import os
import time
import json
import glob
import shutil
import base64
from openai import OpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Carrega as variáveis salvas no arquivo oculto .env
load_dotenv()


# =====================================================================
# 1. CONFIGURAÇÃO INICIAL E CONTRATO DE DADOS DE TI (OPENROUTER)
# =====================================================================

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("GRAPHRAG_API_KEY")
)

MODELO = "google/gemini-2.5-flash-lite"  # modelo multimodal: aceita texto, PDF, áudio e vídeo

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

RESPONSE_FORMAT_ESTRUTURADO = {
    "type": "json_schema",
    "json_schema": {
        "name": "extracao_grafo_ti",
        "strict": True,
        "schema": ExtracaoGrafoTI.model_json_schema(),
    },
}

PROMPT_COMANDO = (
    "Você é um engenheiro de sistemas sênior e professor especialista em concursos de Tecnologia da Informação (TI). "
    "Analise minuciosamente o material fornecido (pode ser um PDF, um áudio, um vídeo ou texto). "
    "Identifique todas as tecnologias, protocolos, arquiteturas, algoritmos, pegadinhas de bancas e questões presentes. "
    "Gere uma lista estrita de Nós e Relacionamentos em JSON."
)

MIME_POR_EXTENSAO = {
    ".pdf": "application/pdf",
    ".mp3": "audio/mp3",
    ".wav": "audio/wav",
    ".mp4": "video/mp4",
}

FORMATO_AUDIO_POR_EXTENSAO = {".mp3": "mp3", ".wav": "wav"}


# =====================================================================
# 2. MONTAGEM DO CONTEÚDO MULTIMODAL (PDF / ÁUDIO / VÍDEO)
# =====================================================================

def _codificar_base64(caminho_arquivo: str) -> str:
    with open(caminho_arquivo, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def montar_bloco_arquivo(caminho_arquivo: str, extensao: str) -> dict:
    """Monta o bloco de conteúdo multimodal certo para cada tipo de arquivo,
    conforme o formato exigido pela OpenRouter."""
    nome_arquivo = os.path.basename(caminho_arquivo)
    dados_base64 = _codificar_base64(caminho_arquivo)
    mime_type = MIME_POR_EXTENSAO[extensao]

    if extensao == ".pdf":
        return {
            "type": "file",
            "file": {
                "filename": nome_arquivo,
                "file_data": f"data:{mime_type};base64,{dados_base64}",
            },
        }

    if extensao in (".mp3", ".wav"):
        return {
            "type": "input_audio",
            "input_audio": {
                "data": dados_base64,
                "format": FORMATO_AUDIO_POR_EXTENSAO[extensao],
            },
        }

    if extensao == ".mp4":
        return {
            "type": "video_url",
            "video_url": {
                "url": f"data:{mime_type};base64,{dados_base64}",
            },
        }

    raise ValueError(f"Extensão não suportada: {extensao}")


# =====================================================================
# 3. CHAMADAS AO MODELO (EXTRAÇÃO ESTRUTURADA)
# =====================================================================

def extrair_grafo_de_arquivo(caminho_arquivo: str, extensao: str, descricao_materia: str) -> str:
    """Manda o arquivo (PDF/áudio/vídeo) DIRETO pro modelo multimodal."""
    bloco_arquivo = montar_bloco_arquivo(caminho_arquivo, extensao)

    response = client.chat.completions.create(
        model=MODELO,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{PROMPT_COMANDO}\nContexto: {descricao_materia}"},
                    bloco_arquivo,
                ],
            }
        ],
        response_format=RESPONSE_FORMAT_ESTRUTURADO,
        temperature=0.1,
    )

    print("        [Controle de Taxa] Pausando por 20 segundos para preservar os créditos...")
    time.sleep(20)
    return response.choices[0].message.content


def extrair_grafo_de_youtube(url: str, descricao_materia: str) -> str:
    """Manda o LINK do YouTube direto pro modelo, via video_url.
    Só funciona com provedores que suportam vídeo por URL (ex: Google AI Studio).
    """
    response = client.chat.completions.create(
        model=MODELO,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{PROMPT_COMANDO}\nContexto: {descricao_materia}"},
                    {"type": "video_url", "video_url": {"url": url}},
                ],
            }
        ],
        response_format=RESPONSE_FORMAT_ESTRUTURADO,
        temperature=0.1,
        extra_body={"provider": {"only": ["google-ai-studio"]}},  # exige provedor com suporte a link do YouTube
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
# 4. EXECUTORES AUTOMÁTICOS (ARQUIVOS E LINKS)
# =====================================================================

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
            extensao = extensao.lower()

            print(f"\n🎬 Processando Arquivo Local: {nome_arquivo}")
            try:
                json_extraido = extrair_grafo_de_arquivo(caminho_completo, extensao, f"Mídia local {nome_puro}")
                salvar_dados_para_graphrag(json_extraido, nome_base=nome_puro)
                shutil.move(caminho_completo, os.path.join(pasta_destino, nome_arquivo))
                print(f"📦 [Sucesso] Arquivo original movido para: {pasta_destino}")
            except Exception as e:
                print(f"❌ [Erro] Falha no arquivo {nome_arquivo}: {e}")
                if "429" in str(e) or "quota" in str(e).lower() or "credit" in str(e).lower():
                    return False
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
            json_extraido = extrair_grafo_de_youtube(url, f"Videoaula extraída do link {url}")
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
# 5. VALIDAÇÃO DE ENTRADA DO ECOSSISTEMA
# =====================================================================
if __name__ == "__main__":
    print("=== INICIANDO ECOSSISTEMA TOTAL DE TI VIA OPENROUTER (MULTIMODAL DIRETO) ===")

    extensoes = ["*.pdf", "*.mp4", "*.mp3", "*.wav"]
    arquivos_locais = []
    for ext in extensoes:
        arquivos_locais.extend(glob.glob(ext))

    tem_links = False
    if os.path.exists("links.txt"):
        with open("links.txt", "r", encoding="utf-8") as f:
            linhas_links = [l.strip() for l in f if l.strip() and not l.strip().startswith("#")]
            if linhas_links:
                tem_links = True

    if not arquivos_locais and not tem_links:
        print("sem links ou arquivos para processar")
        exit(0)

    sucesso_locais = processar_arquivos_locais_automatico()
    if sucesso_locais:
        processar_links_txt_automatico()

    print("\n=== PIPELINE DE EXTRAÇÃO CONCLUÍDO ===")