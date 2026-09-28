import pymupdf as fitz
import pdfplumber
import duckdb
import pandas as pd
import json
import os
import base64
from openai import OpenAI
from dotenv import load_dotenv  

load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

# ----------------------------------------------------
# CONFIGURAÇÃO DE PASTAS DO ECOSSISTEMA
# ----------------------------------------------------
PASTA_PDFS_ENTRADA = "meus_pdfs"
PASTA_IMAGENS_EXTRAIDAS = "imagens_extraidas"
PASTA_GRAPH_RAG = "dados_entrada"

os.makedirs(PASTA_PDFS_ENTRADA, exist_ok=True)
os.makedirs(PASTA_IMAGENS_EXTRAIDAS, exist_ok=True)
os.makedirs(PASTA_GRAPH_RAG, exist_ok=True)

def encode_image_to_base64(caminho_imagem):
    """Converte o arquivo físico da imagem em Base64 exigido pela API"""
    with open(caminho_imagem, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def descrever_imagem_com_gemini_openrouter(caminho_imagem):
    """Envia a figura para o Gemini através do OpenRouter tratando retornos variados da API"""
    if not os.environ.get("GRAPHRAG_API_KEY"):
        print("   ⚠️ [Aviso] GRAPHRAG_API_KEY não localizada nas variáveis de ambiente.")
        return "Descrição indisponível: Chave de API do OpenRouter ausente."
        
    try:
        client = OpenAI(
            base_url="https://openrouter.ai",
            api_key=os.environ.get("GRAPHRAG_API_KEY")
        )
        
        base64_image = encode_image_to_base64(caminho_imagem)
        extensao = os.path.splitext(caminho_imagem)[-1].replace(".", "").lower()
        if extensao == "jpg": extensao = "jpeg"

        resposta = client.chat.completions.create(
            model="google/gemini-2.5-flash",
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Descreva detalhadamente o que está nesta imagem extraída de um documento técnico ou legal. Se for um gráfico, descreva os eixos e liste os dados. Se for un organograma ou fluxograma, detalhe as etapas sequenciais e as conexões textuais internas."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{extensao};base64,{base64_image}"
                        }
                    }
                ]
            }]
        )
        
        # 🆕 CORREÇÃO DE ARQUITETURA DE RETORNO:
        # Se a API responder como String direta (caso comum em alguns gateways do OpenRouter)
        if isinstance(resposta, str):
            return resposta.strip()
            
        # Se a API responder como dicionário clássico Python
        if isinstance(resposta, dict):
            if 'choices' in resposta and len(resposta['choices']) > 0:
                return resposta['choices'][0]['message']['content'].strip()
            return str(resposta)
            
        # Se a API responder como o objeto tradicional da SDK OpenAI
        if hasattr(resposta, 'choices') and len(resposta.choices) > 0:
            return resposta.choices[0].message.content.strip()
            
        return "Não foi possível extrair um formato de texto válido da resposta da API."
        
    except Exception as e:
        print(f"   ⚠️ Falha ao consultar o OpenRouter: {e}")
        return f"Descrição visual indisponível devido a um erro no OpenRouter: {e}"


def inicializar_banco():
    """Cria a estrutura de tabelas relacionais do DuckDB caso não existam"""
    con = duckdb.connect("banco_pdf.db")
    con.execute("""
        CREATE TABLE IF NOT EXISTS texto_paginas (
            documento VARCHAR, 
            pagina INTEGER, 
            conteudo_texto TEXT
        );
        CREATE TABLE IF NOT EXISTS tabelas_pdf (
            documento VARCHAR, 
            pagina INTEGER, 
            indice_tabela INTEGER, 
            conteudo_tabela_json TEXT
        );
        CREATE TABLE IF NOT EXISTS figuras_pdf (
            documento VARCHAR, 
            pagina INTEGER, 
            indice_figura INTEGER, 
            caminho_local TEXT, 
            formato VARCHAR, 
            descricao_visual TEXT
        );
    """)
    con.close()

def processar_pdf_com_upsert(caminho_pdf):
    """Executa a ingestão síncrona limpando dados antigos se o arquivo já existir (Upsert)"""
    nome_pdf = os.path.basename(caminho_pdf)
    print(f"\n🎬 Iniciando Ingestão de Dados: {nome_pdf}")
    
    con = duckdb.connect("banco_pdf.db")
    
    # Mecânica do Upsert: Limpa dados antigos APENAS deste documento específico
    con.execute("DELETE FROM texto_paginas WHERE documento = ?;", [nome_pdf])
    con.execute("DELETE FROM tabelas_pdf WHERE documento = ?;", [nome_pdf])
    con.execute("DELETE FROM figuras_pdf WHERE documento = ?;", [nome_pdf])
    
    doc_fitz = fitz.open(caminho_pdf)
    pdf_plumber = pdfplumber.open(caminho_pdf)
    total_paginas = len(doc_fitz)
    
    for idx_pag in range(total_paginas):
        num_pag_real = idx_pag + 1
        print(f"📖 Extraindo Elementos - Página {num_pag_real}/{total_paginas}...")
        
        # 1. Extração de Texto Puro (PyMuPDF)
        pagina_fitz = doc_fitz[idx_pag]
        texto_puro = pagina_fitz.get_text().strip()
        if texto_puro:
            con.execute("INSERT INTO texto_paginas VALUES (?, ?, ?)", (nome_pdf, num_pag_real, texto_puro))
            
        # 2. Extração de Tabelas Estruturadas (pdfplumber)
        pagina_plumber = pdf_plumber.pages[idx_pag]
        tabelas = pagina_plumber.extract_tables()
        for idx_tab, tabela in enumerate(tabelas):
            if tabela:
                tabela_json = json.dumps(tabela, ensure_ascii=False)
                con.execute("INSERT INTO tabelas_pdf VALUES (?, ?, ?, ?)", (nome_pdf, num_pag_real, idx_tab, tabela_json))
                
        # 3. Extração de Imagens + Visão Computacional (OpenRouter)
        lista_imagens = pagina_fitz.get_images(full=True)
        for idx_img, img in enumerate(lista_imagens):
            xref = img[0]
            base_image = doc_fitz.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            
            # Sanitiza o nome do arquivo físico para evitar quebras em disco
            nome_limpo_pdf = "".join([c if c.isalnum() else "_" for c in nome_pdf])
            nome_arquivo_img = f"img_{nome_limpo_pdf}_pag_{num_pag_real}_{idx_img}.{image_ext}"
            caminho_salvar_img = os.path.join(PASTA_IMAGENS_EXTRAIDAS, nome_arquivo_img)
            
            with open(caminho_salvar_img, "wb") as f_img:
                f_img.write(image_bytes)
                
            # Aciona a descrição por Inteligência Artificial
            descricao_gemini = "Arquivo corrompido"
            if os.path.exists(caminho_salvar_img) and os.path.getsize(caminho_salvar_img) > 0:
                descricao_gemini = descrever_imagem_com_gemini_openrouter(caminho_salvar_img)
            
            con.execute("INSERT INTO figuras_pdf VALUES (?, ?, ?, ?, ?, ?)", 
                        (nome_pdf, num_pag_real, idx_img, caminho_salvar_img, image_ext, descricao_gemini))
            
    doc_fitz.close()
    pdf_plumber.close()
    con.close()
    print(f"✅ Ingestão concluída com sucesso para o documento '{nome_pdf}'.")

def exportar_para_graphrag_jsonl():
    """Lê o banco de dados acumulado e gera saídas no formato JSONL estruturado de alta fidelidade"""
    print("\n🦆 [DuckDB] Iniciando a exportação geral para o padrão JSONL...")
    con = duckdb.connect("banco_pdf.db")
    
    documentos = [r[0] for r in con.execute("SELECT DISTINCT documento FROM texto_paginas UNION SELECT DISTINCT documento FROM tabelas_pdf UNION SELECT DISTINCT documento FROM figuras_pdf").fetchall() if r[0]]
    
    for doc in documentos:
        caminho_jsonl = os.path.join(PASTA_GRAPH_RAG, f"dados_{doc}.jsonl")
        
        with open(caminho_jsonl, "w", encoding="utf-8") as f_out:
            # 1. Exportando Textos por Página
            df_t = con.execute("SELECT pagina, conteudo_texto FROM texto_paginas WHERE documento = ?", [doc]).df()
            for _, row in df_t.iterrows():
                obj = {
                    "id": f"{doc}_pag_{row['pagina']}_texto",
                    "title": f"Texto Normativo - {doc} - Página {row['pagina']}",
                    "text": str(row['conteudo_texto'])
                }
                f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")
                
            # 2. Exportando Tabelas Individuais Blindadas por ID
            df_tab = con.execute("SELECT pagina, indice_tabela, conteudo_tabela_json FROM tabelas_pdf WHERE documento = ?", [doc]).df()
            for _, row in df_tab.iterrows():
                matriz = json.loads(row['conteudo_tabela_json'])
                texto_tabela = f"Dados estruturados da Tabela {row['indice_tabela']} na página {row['pagina']}:\n"
                for idx, linha in enumerate(matriz):
                    texto_tabela += f"Linha {idx+1}: {', '.join([str(v) for v in linha if v is not None])}\n"
                
                obj = {
                    "id": f"{doc}_pag_{row['pagina']}_tab_{row['indice_tabela']}",
                    "title": f"Tabela {row['indice_tabela']} - {doc} - Página {row['pagina']}",
                    "text": texto_tabela.strip()
                }
                f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")
                
            # 3. Exportando Imagens com Descrição de Visão
            df_fig = con.execute("SELECT pagina, indice_figura, caminho_local, descricao_visual FROM figuras_pdf WHERE documento = ?", [doc]).df()
            for _, row in df_fig.iterrows():
                obj = {
                    "id": f"{doc}_pag_{row['pagina']}_fig_{row['indice_figura']}",
                    "title": f"Elemento Gráfico {row['indice_figura']} - {doc} - Página {row['pagina']}",
                    "text": f"Análise visual da imagem salva localmente em '{row['caminho_local']}': {row['descricao_visual']}"
                }
                f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")
                
        print(f"   ✅ Arquivo JSONL estruturado pronto: {caminho_jsonl}")
        
    con.close()
    print("🎉 Exportação concluída! Os dados estão prontos para a criação automática da ontologia.")

if __name__ == "__main__":
    inicializar_banco()
    
    # Varre a pasta de entrada mapeando arquivos PDF
    arquivos_encontrados = [f for f in os.listdir(PASTA_PDFS_ENTRADA) if f.endswith(".pdf")]
    
if not arquivos_encontrados:
    print(f"⚠️ Pasta vazia! Jogue seus arquivos PDF na pasta operacional: '{PASTA_PDFS_ENTRADA}'")
else:
    print(f"📂 Encontrados {len(arquivos_encontrados)} arquivos PDF para processar.")
    for arquivo in arquivos_encontrados:
        caminho_completo_pdf = os.path.join(PASTA_PDFS_ENTRADA, arquivo)
        processar_pdf_com_upsert(caminho_completo_pdf)
    exportar_para_graphrag_jsonl()
