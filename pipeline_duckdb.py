import pymupdf as fitz
import pdfplumber
import duckdb
import pandas as pd
import json
import os
import base64
import hashlib
import time
from openai import OpenAI
from dotenv import load_dotenv

# ----------------------------------------------------
# CAMINHOS: sempre relativos à pasta deste script,
# não à pasta de onde você rodou o comando
# ----------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

PASTA_PDFS_ENTRADA = os.path.join(BASE_DIR, "meus_pdfs")
PASTA_IMAGENS_EXTRAIDAS = os.path.join(BASE_DIR, "imagens_extraidas")
PASTA_GRAPH_RAG = os.path.join(BASE_DIR, "dados_entrada")
CAMINHO_BANCO = os.path.join(BASE_DIR, "banco_pdf.db")

os.makedirs(PASTA_PDFS_ENTRADA, exist_ok=True)
os.makedirs(PASTA_IMAGENS_EXTRAIDAS, exist_ok=True)
os.makedirs(PASTA_GRAPH_RAG, exist_ok=True)

# ----------------------------------------------------
# PARÂMETROS AJUSTÁVEIS
# ----------------------------------------------------
MODELO_VISAO = "google/gemini-2.5-flash"
OPENROUTER_URL = "https://openrouter.ai/api/v1"
TAMANHO_MIN_IMAGEM = 150          # ignora imagens menores (logos, ícones, linhas)
RENDERIZAR_PAGINAS_VETORIAIS = True   # descreve páginas com gráficos/fluxogramas vetoriais
LIMIAR_DESENHOS_VETORIAIS = 20    # nº mínimo de traços para considerar a página "desenhada"
TENTATIVAS_API = 2

PROMPT_VISAO = (
    "Você está analisando uma imagem extraída de um documento técnico ou legal. "
    "1) TRANSCREVA literalmente todo o texto visível na imagem, preservando a ordem. "
    "2) Depois descreva a estrutura: se for gráfico, descreva os eixos e liste os dados; "
    "se for fluxograma ou organograma, detalhe as etapas sequenciais e as conexões; "
    "se for tabela, reproduza linhas e colunas. "
    "Se a imagem for apenas decorativa (logo, moldura, ícone), responda exatamente: DECORATIVA. "
    "Responda em português."
)

FORMATOS_ACEITOS = {"png", "jpeg", "jpg", "webp", "gif"}


def encode_image_to_base64(caminho_imagem):
    """Converte o arquivo físico da imagem em Base64 exigido pela API"""
    with open(caminho_imagem, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def descrever_imagem_com_gemini_openrouter(caminho_imagem):
    """
    Envia a figura ao Gemini via OpenRouter.
    Retorna o texto da descrição, ou None se falhar / for decorativa.
    Nunca retorna mensagem de erro como se fosse conteúdo (para não poluir o grafo).
    """
    chave = os.environ.get("GRAPHRAG_API_KEY")
    if not chave:
        print("   ⚠️ [Aviso] GRAPHRAG_API_KEY não localizada. Imagem ignorada.")
        return None

    client = OpenAI(base_url=OPENROUTER_URL, api_key=chave)

    base64_image = encode_image_to_base64(caminho_imagem)
    extensao = os.path.splitext(caminho_imagem)[-1].replace(".", "").lower()
    if extensao == "jpg":
        extensao = "jpeg"

    for tentativa in range(1, TENTATIVAS_API + 1):
        try:
            resposta = client.chat.completions.create(
                model=MODELO_VISAO,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": PROMPT_VISAO},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/{extensao};base64,{base64_image}"},
                        },
                    ],
                }],
            )
            texto = (resposta.choices[0].message.content or "").strip()
            if not texto:
                raise ValueError("resposta vazia")
            if texto.upper().startswith("DECORATIVA"):
                return None
            return texto
        except Exception as e:
            print(f"   ⚠️ Falha ao consultar o OpenRouter (tentativa {tentativa}/{TENTATIVAS_API}): {e}")
            time.sleep(2 * tentativa)

    return None


def inicializar_banco():
    """Cria a estrutura de tabelas relacionais do DuckDB caso não existam"""
    con = duckdb.connect(CAMINHO_BANCO)
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


def extrair_bytes_imagem(doc_fitz, xref):
    """
    Extrai a imagem pelo xref. Converte para PNG quando o formato original
    não é aceito pela API (ex.: jpx, jb2, CMYK).
    Retorna (bytes, ext, largura, altura).
    """
    base_image = doc_fitz.extract_image(xref)
    largura, altura = base_image["width"], base_image["height"]
    ext = base_image["ext"].lower()
    dados = base_image["image"]

    if ext not in FORMATOS_ACEITOS:
        pix = fitz.Pixmap(doc_fitz, xref)
        if pix.n - pix.alpha >= 4:
            pix = fitz.Pixmap(fitz.csRGB, pix)
        dados = pix.tobytes("png")
        ext = "png"

    return dados, ext, largura, altura


def processar_pdf_com_upsert(caminho_pdf):
    """Executa a ingestão limpando dados antigos se o arquivo já existir (Upsert)"""
    nome_pdf = os.path.basename(caminho_pdf)
    nome_limpo_pdf = "".join(c if c.isalnum() else "_" for c in nome_pdf)
    print(f"\n🎬 Iniciando Ingestão de Dados: {nome_pdf}")

    con = duckdb.connect(CAMINHO_BANCO)

    con.execute("DELETE FROM texto_paginas WHERE documento = ?;", [nome_pdf])
    con.execute("DELETE FROM tabelas_pdf WHERE documento = ?;", [nome_pdf])
    con.execute("DELETE FROM figuras_pdf WHERE documento = ?;", [nome_pdf])

    doc_fitz = fitz.open(caminho_pdf)
    pdf_plumber = pdfplumber.open(caminho_pdf)
    total_paginas = len(doc_fitz)

    hashes_vistos = set()  # evita descrever a mesma imagem (ex.: logo) várias vezes
    total_figuras = 0

    for idx_pag in range(total_paginas):
        num_pag_real = idx_pag + 1
        print(f"📖 Extraindo Elementos - Página {num_pag_real}/{total_paginas}...")

        # 1. Texto puro (PyMuPDF)
        pagina_fitz = doc_fitz[idx_pag]
        texto_puro = pagina_fitz.get_text().strip()
        if texto_puro:
            con.execute("INSERT INTO texto_paginas VALUES (?, ?, ?)",
                        (nome_pdf, num_pag_real, texto_puro))

        # 2. Tabelas (pdfplumber)
        try:
            tabelas = pdf_plumber.pages[idx_pag].extract_tables()
        except Exception as e:
            print(f"   ⚠️ Falha ao extrair tabelas da página {num_pag_real}: {e}")
            tabelas = []
        for idx_tab, tabela in enumerate(tabelas):
            if tabela:
                tabela_json = json.dumps(tabela, ensure_ascii=False)
                con.execute("INSERT INTO tabelas_pdf VALUES (?, ?, ?, ?)",
                            (nome_pdf, num_pag_real, idx_tab, tabela_json))

        # 3. Imagens embutidas + descrição por visão computacional
        lista_imagens = pagina_fitz.get_images(full=True)
        figuras_da_pagina = 0

        for idx_img, img in enumerate(lista_imagens):
            xref = img[0]
            try:
                dados, ext, largura, altura = extrair_bytes_imagem(doc_fitz, xref)
            except Exception as e:
                print(f"   ⚠️ Não foi possível extrair a imagem xref={xref}: {e}")
                continue

            if largura < TAMANHO_MIN_IMAGEM or altura < TAMANHO_MIN_IMAGEM:
                continue

            h = hashlib.md5(dados).hexdigest()
            if h in hashes_vistos:
                continue
            hashes_vistos.add(h)

            nome_arquivo_img = f"img_{nome_limpo_pdf}_pag_{num_pag_real}_{idx_img}.{ext}"
            caminho_img = os.path.join(PASTA_IMAGENS_EXTRAIDAS, nome_arquivo_img)
            with open(caminho_img, "wb") as f_img:
                f_img.write(dados)

            descricao = descrever_imagem_com_gemini_openrouter(caminho_img)
            if descricao:
                con.execute("INSERT INTO figuras_pdf VALUES (?, ?, ?, ?, ?, ?)",
                            (nome_pdf, num_pag_real, idx_img, caminho_img, ext, descricao))
                figuras_da_pagina += 1
                total_figuras += 1

        # 4. Gráficos/fluxogramas vetoriais: renderiza a página inteira
        if RENDERIZAR_PAGINAS_VETORIAIS and figuras_da_pagina == 0:
            try:
                n_desenhos = len(pagina_fitz.get_drawings())
            except Exception:
                n_desenhos = 0
            if n_desenhos > LIMIAR_DESENHOS_VETORIAIS:
                pix = pagina_fitz.get_pixmap(dpi=120)
                caminho_img = os.path.join(
                    PASTA_IMAGENS_EXTRAIDAS,
                    f"img_{nome_limpo_pdf}_pag_{num_pag_real}_render.png")
                pix.save(caminho_img)
                descricao = descrever_imagem_com_gemini_openrouter(caminho_img)
                if descricao:
                    con.execute("INSERT INTO figuras_pdf VALUES (?, ?, ?, ?, ?, ?)",
                                (nome_pdf, num_pag_real, 900, caminho_img, "png", descricao))
                    total_figuras += 1

    doc_fitz.close()
    pdf_plumber.close()
    con.close()
    print(f"✅ Ingestão concluída para '{nome_pdf}' ({total_figuras} figuras descritas).")


def exportar_para_graphrag_jsonl():
    """Lê o banco acumulado e gera um .jsonl por documento"""
    print("\n🦆 [DuckDB] Iniciando a exportação geral para o padrão JSONL...")
    con = duckdb.connect(CAMINHO_BANCO)

    documentos = [
        r[0] for r in con.execute(
            "SELECT DISTINCT documento FROM texto_paginas "
            "UNION SELECT DISTINCT documento FROM tabelas_pdf "
            "UNION SELECT DISTINCT documento FROM figuras_pdf"
        ).fetchall() if r[0]
    ]

    for doc in documentos:
        caminho_jsonl = os.path.join(PASTA_GRAPH_RAG, f"dados_{doc}.jsonl")

        with open(caminho_jsonl, "w", encoding="utf-8") as f_out:
            # 1. Texto por página
            df_t = con.execute(
                "SELECT pagina, conteudo_texto FROM texto_paginas WHERE documento = ? ORDER BY pagina",
                [doc]).df()
            for _, row in df_t.iterrows():
                obj = {
                    "id": f"{doc}_pag_{row['pagina']}_texto",
                    "title": f"Texto Normativo - {doc} - Página {row['pagina']}",
                    "text": str(row["conteudo_texto"]),
                }
                f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")

            # 2. Tabelas
            df_tab = con.execute(
                "SELECT pagina, indice_tabela, conteudo_tabela_json FROM tabelas_pdf "
                "WHERE documento = ? ORDER BY pagina, indice_tabela",
                [doc]).df()
            for _, row in df_tab.iterrows():
                matriz = json.loads(row["conteudo_tabela_json"])
                texto_tabela = f"Dados estruturados da Tabela {row['indice_tabela']} na página {row['pagina']}:\n"
                for idx, linha in enumerate(matriz):
                    celulas = [str(v).replace("\n", " ") for v in linha if v is not None]
                    texto_tabela += f"Linha {idx + 1}: {', '.join(celulas)}\n"
                obj = {
                    "id": f"{doc}_pag_{row['pagina']}_tab_{row['indice_tabela']}",
                    "title": f"Tabela {row['indice_tabela']} - {doc} - Página {row['pagina']}",
                    "text": texto_tabela.strip(),
                }
                f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")

            # 3. Figuras com a descrição/transcrição da visão
            df_fig = con.execute(
                "SELECT pagina, indice_figura, caminho_local, descricao_visual FROM figuras_pdf "
                "WHERE documento = ? ORDER BY pagina, indice_figura",
                [doc]).df()
            for _, row in df_fig.iterrows():
                obj = {
                    "id": f"{doc}_pag_{row['pagina']}_fig_{row['indice_figura']}",
                    "title": f"Elemento Gráfico {row['indice_figura']} - {doc} - Página {row['pagina']}",
                    "text": (f"Análise visual de figura da página {row['pagina']} "
                             f"do documento {doc} "
                             f"(arquivo {os.path.basename(row['caminho_local'])}):\n"
                             f"{row['descricao_visual']}"),
                }
                f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")

        print(f"   ✅ Arquivo JSONL pronto: {caminho_jsonl}")

    con.close()
    print("🎉 Exportação concluída! Os dados estão prontos para o GraphRAG.")


if __name__ == "__main__":
    inicializar_banco()

    arquivos_encontrados = [f for f in os.listdir(PASTA_PDFS_ENTRADA) if f.lower().endswith(".pdf")]

    if not arquivos_encontrados:
        print(f"⚠️ Pasta vazia! Coloque seus PDFs em: '{PASTA_PDFS_ENTRADA}'")
    else:
        print(f"📂 Encontrados {len(arquivos_encontrados)} arquivos PDF para processar.")
        for arquivo in arquivos_encontrados:
            processar_pdf_com_upsert(os.path.join(PASTA_PDFS_ENTRADA, arquivo))
        exportar_para_graphrag_jsonl()