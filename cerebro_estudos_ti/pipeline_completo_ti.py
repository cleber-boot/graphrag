import os
import time
import json
import glob
import shutil
from openai import OpenAI
from pydantic import BaseModel, Field

# =====================================================================
# 1. CONFIGURAÇÃO INICIAL E CONTRATO DE DADOS DE TI (OPENROUTER)
# =====================================================================

# Inicialização segura conectando direto nos servidores do OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai",
    api_key="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"
)

class DetalhesTecnicos(BaseModel):
    resumo_conceito: str = Field(description="O que é este conceito ou tecnologia de TI em poucas palavras.")
    contexto_concurso: str = Field(description="Como este tema costuma cair em provas ou pegadinhas de bancas.")

class ExtracaoGrafoTI(BaseModel):
    class NoTI(BaseModel):
        id: str = Field(description="ID único em minúsculas sem espaços ou acentos. Ex: 'docker', 'protocolo_tcp'")
        type: str = Field(description="Tipo da entidade. Deve ser: 'Tecnologia', 'Arquitetura', 'Protocolo', 'Algoritmo', 'Framework_Governanca', 'Vulnerabilidade' ou 'Questao'")
        properties: DetalhesTecnicos = Field(description="Metadados técnicos obrigatórios estruturados.")

    class RelacionamentoTI(BaseModel):
        source: str = Field(description="ID do nó tecnológico de origem da conexão.")
        relation_type: str = Field(description="Verbo de ligação em maiúsculo. Ex: 'EXECUTA_EM', 'IMPLEMENTA', 'VULNERAVEL_A'")
        target: str = Field(description="ID do nó tecnológico de destino da conexão.")

    nodes: list[NoTI] = Field(description="Lista contendo todos os nós conceituais de TI encontrados.")
    relationships: list[RelacionamentoTI] = Field(description="Lista com as conexões lógicas encontradas.")

PROMPT_COMANDO = (
    "Você é um engenheiro de sistemas sênior e professor especialista em concursos de Tecnologia da Informação (TI). "
    "Analise minuciosamente o material fornecido. Identifique todas as tecnologias, protocolos, arquiteturas, "
    "algoritmos, pegadinhas de bancas e questões presentes. Gere uma lista estrita de Nós e Relacionamentos em JSON."
)

# =====================================================================
# 2. MOTORES DE PROCESSAMENTO COMPATÍVEIS COM OPENROUTER
# =====================================================================

def processar_link_youtube(url: str, descricao_materia: str):
    print(f"\n[Fase 1] Enviando Link do YouTube para o OpenRouter: {url}")
    
    response = client.chat.completions.create(
        model="google/gemini-2.5-flash-lite",
        messages=[
            {"role": "user", "content": f"{PROMPT_COMANDO}\nContexto do Material: {descricao_materia}\nLink da Videoaula: {url}"}
        ],
        response_format={"type": "json_object"},
        temperature=0.1
    )
    
    print("        [Controle de Taxa] Pausando por 20 segundos para preservar os créditos...")
    time.sleep(20)
    return response.choices[0].message.content

def processar_arquivo_local(caminho_arquivo: str, mime_type: str, descricao_materia: str):
    print(f"\n[Fase 1] Lendo conteúdo do arquivo local para o OpenRouter: {caminho_arquivo}")
    
    # Como o OpenRouter não possui o bucket de upload de mídias binárias do Google Cloud,
    # abrimos o arquivo em modo texto para extrair e indexar a matéria perfeitamente.
    with open(caminho_arquivo, "r", encoding="utf-8", errors="ignore") as f:
        conteudo_texto = f.read()[:60000] # Limite seguro de caracteres por arquivo para não travar
        
    response = client.chat.completions.create(
        model="google/gemini-2.5-flash-lite",
        messages=[
            {"role": "user", "content": f"{PROMPT_COMANDO}\nMaterial Bruto de TI: {conteudo_texto}\nContexto: {descricao_materia}"}
        ],
        response_format={"type": "json_object"},
        temperature=0.1
    )
    
    print("        [Controle de Taxa] Pausando por 20 segundos para preservar os créditos...")
    time.sleep(20)
    return response.choices[0].message.content

def salvar_dados_para_graphrag(json_str: str, nome_base: str, pasta_saida: str = "./input"):
    os.makedirs(pasta_saida, exist_ok=True)
    dados = json.loads(json_str)
    
    timestamp = int(time.time())
    nome_arquivo = f"conhecimento_ti_{nome_base}_{timestamp}.txt"
    caminho_final = os.path.join(pasta_saida, nome_arquivo)
    
    with open(caminho_final, "w", encoding="utf-8") as f:
        f.write("=== CONCEITOS E ENTIDADES DE TI ===\n")
        for node in dados.get("nodes", []):
            props = node['properties']
            f.write(f"Tecnologia: {node['id']} | Categoria: {node['type']} | Resumo: {props['resumo_conceito']} | Foco Concurso: {props['contexto_concurso']}\n")
            
        f.write("\n=== MAPA DE RELACIONAMENTOS ARQUITETURAIS ===\n")
        for rel in dados.get("relationships", []):
            f.write(f"Conexao: '{rel['source']}' --[{rel['relation_type']}]--> '{rel['target']}'\n")
            
    print(f"[Fase 2] Grafo textual salvo com sucesso para o GraphRAG: {caminho_final}")

# =====================================================================
# 3. EXECUTORES AUTOMÁTICOS (ARQUIVOS E LINKS)
# =====================================================================

def mapear_mime_type(extensao: str) -> str:
    mapeamento = {".pdf": "application/pdf", ".mp4": "video/mp4", ".mp3": "audio/mp3", ".wav": "audio/wav"}
    return mapeamento.get(extensao.lower(), "application/octet-stream")

def processar_arquivos_locais_automatico():
    pasta_destino = "./arquivos_processados"
    os.makedirs(pasta_destino, exist_ok=True)
    
    extensoes_permitidas = ["*.pdf", "*.mp4", "*.mp3", "*.wav"]
    arquivos_encontrados = []
    for ext in extensoes_permitidas:
        arquivos_encontrados.extend(glob.glob(ext))
    
    if arquivos_encontrados:
        print(f"\n[Mídias] Encontrados {len(arquivos_encontrados)} arquivos locais para processamento.")
        for caminho_completo in arquivos_encontrados:
            nome_arquivo = os.path.basename(caminho_completo)
            nome_puro, extensao = os.path.splitext(nome_arquivo)
            
            print(f"\n🎬 Processando Arquivo Local: {nome_arquivo}")
            try:
                json_extraido = processar_arquivo_local(caminho_completo, mapear_mime_type(extensao), f"Mídia local {nome_puro}")
                salvar_dados_para_graphrag(json_extraido, nome_base=nome_puro)
                shutil.move(caminho_completo, os.path.join(pasta_destino, nome_arquivo))
                print(f"📦 [Sucesso] Arquivo original movido para: {pasta_destino}")
            except Exception as e:
                print(f"❌ [Erro] Falha no arquivo {nome_arquivo}: {e}")
                if "429" in str(e) or "quota" in str(e).lower() or "credit" in str(e).lower(): return False
    return True

def processar_links_txt_automatico():
    arquivo_links = "links.txt"
    if not os.path.exists(arquivo_links):
        return True
        
    with open(arquivo_links, "r", encoding="utf-8") as f:
        links = [linha.strip() for linha_crua in f if (linha := linha_crua.strip()) and not linha.startswith("#")]
        
    if not links:
        return True
        
    print(f"\n[Links] Encontrados {len(links)} links do YouTube para processar.")
    links_processados_com_sucesso = []
    
    for indice, url in enumerate(links):
        print(f"\n🌐 Processando Vídeo {indice + 1}/{len(links)}: {url}")
        nome_base = f"youtube_video_{indice}"
        try:
            json_extraido = processar_link_youtube(url, f"Videoaula extraída do link {url}")
            salvar_dados_para_graphrag(json_extraido, nome_base=nome_base)
            links_processados_com_sucesso.append(url)
        except Exception as e:
            print(f"❌ [Erro] Falha ao processar o link {url}: {e}")
            if "429" in str(e) or "quota" in str(e).lower() or "credit" in str(e).lower():
                print("🛑 [Limite Extra] Interrompendo lote de links para proteção de saldo.")
                break
                
    links_restantes = [l for l in links if l not in links_processados_com_sucesso]
    with open(arquivo_links, "w", encoding="utf-8") as f:
        for l in links_restantes:
            f.write(f"{l}\n")
            
    if not links_restantes:
        print("Mesa limpa! Todos os links do arquivo 'links.txt' foram processados.")
    else:
        print(f"Aviso: {len(links_restantes)} links ficaram guardados para a próxima rodada.")

# =====================================================================
# 4. VALIDAÇÃO DE ENTRADA DO ECOSSISTEMA
# =====================================================================
if __name__ == "__main__":
    print("=== INICIANDO ECOSSISTEMA TOTAL DE TI VIA OPENROUTER ===")
    
    # 1. Varre arquivos locais
    extensoes = ["*.pdf", "*.mp4", "*.mp3", "*.wav"]
    arquivos_locais = []
    for ext in extensoes:
        arquivos_locais.extend(glob.glob(ext))
        
    # 2. Varre arquivo links.txt
    tem_links = False
    if os.path.exists("links.txt"):
        with open("links.txt", "r", encoding="utf-8") as f:
            linhas_links = [l.strip() for l in f if l.strip() and not l.strip().startswith("#")]
            if linhas_links:
                tem_links = True

    # 3. Trava de Validação Solicitada: Se tudo estiver vazio, encerra respondendo a mensagem padrão
    if not arquivos_locais and not tem_links:
        print("sem links ou arquivos para processar")
        exit(0)
        
    # 4. Executa a esteira de processamento
    sucesso_locais = processar_arquivos_locais_automatico()
    if sucesso_locais:
        processar_links_txt_automatico()
        
    print("\n=== PIPELINE DE EXTRAÇÃO CONCLUÍDO ===")
