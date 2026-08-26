import os
import time
import json
import glob
import shutil
import base64
from openai import OpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from json_repair import repair_json  # pip install json-repair

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

# Sem isso, a resposta pode ser cortada no meio de uma string do JSON em
# PDFs densos (muitas tabelas/figuras/relações), causando os erros
# "Unterminated string" / "Expecting property name..." no json.loads().
# Suba esse valor se ainda cortar em documentos muito extensos.
MAX_TOKENS_SAIDA = 32000

class MetadadosDocumento(BaseModel):
    titulo: str = Field(description="Nome do Livro/Artigo/Framework de TI abordado no material.")
    tecnologias_citadas: list[str] = Field(description="Lista curta das principais tecnologias citadas no documento inteiro.")
    idioma: str = Field(description="Idioma predominante do material. Ex: 'pt-BR'")


class AtributosTecnicos(BaseModel):
    vantagens: list[str] = Field(description="Pontos fortes / benefícios do conceito.")
    desvantagens: list[str] = Field(description="Limitações, riscos ou trade-offs do conceito.")
    casos_de_uso: list[str] = Field(description="Cenários em que faz sentido aplicar este conceito.")


class ConceitoChave(BaseModel):
    id_entidade: str = Field(description="ID único em minúsculas sem espaços ou acentos. Ex: 'api_gateway', 'idempotencia'")
    termo: str = Field(description="Nome do conceito de TI. Ex: 'API Gateway', 'Idempotência'")
    definicao: str = Field(description="O que é e qual problema de engenharia ele resolve.")
    tipo_entidade: str = Field(description="Categoria. Ex: 'Protocolo', 'Arquitetura', 'Padrão', 'Ferramenta', 'Processo'")
    atributos_tecnicos: AtributosTecnicos


class ValorColuna(BaseModel):
    coluna: str = Field(description="Nome da coluna (deve bater com um item de 'colunas' da tabela).")
    valor: str = Field(description="Valor dessa coluna nesta linha.")


class LinhaTabela(BaseModel):
    id_linha: str
    valores: list[ValorColuna] = Field(description="Pares coluna/valor desta linha, na mesma ordem de 'colunas'.")
    entidades_relacionadas: list[str] = Field(description="IDs de entidades (id_entidade) ligadas a esta linha.")


class Tabela(BaseModel):
    tabela_id: str
    titulo_tabela: str
    descricao_contexto: str = Field(description="Análise comparativa: o que a tabela demonstra tecnicamente.")
    colunas: list[str]
    dados_linhas: list[LinhaTabela]


class FiguraOuGrafico(BaseModel):
    figura_id: str
    titulo_figura: str
    tipo_visual: str = Field(description="Ex: 'Arquitetura de Software', 'Fluxograma de Rede', 'Print de Código'")
    dados_extraidos: str = Field(description="Descrição passo a passo do que a figura mostra.")
    componentes_visuais: list[str] = Field(description="Elementos identificados na figura.")
    entidades_relacionadas: list[str] = Field(description="IDs de entidades (id_entidade) ligadas a esta figura.")


class RelacaoGrafo(BaseModel):
    origem: str = Field(description="id_entidade de origem da conexão.")
    tipo_relacao: str = Field(description="Ex: 'depende_de', 'implementa', 'mitiga_risco', 'substitui', 'comunica_com'")
    destino: str = Field(description="id_entidade ou componente de destino da conexão.")
    descricao_conexao: str = Field(description="Por que o componente A precisa/depende do componente B neste cenário.")


class TextoBase(BaseModel):
    resumo_narrativo: str = Field(description="Resumo denso do funcionamento técnico descrito na seção.")
    conceitos_chave: list[ConceitoChave]


class SecaoConteudo(BaseModel):
    secao_id: str = Field(description="Ex: 'TI_SEC_001'")
    titulo_secao: str = Field(description="Nome do capítulo/seção. Ex: 'Arquitetura de Microserviços'")
    texto_base: TextoBase
    tabelas: list[Tabela]
    figuras_e_graficos: list[FiguraOuGrafico]
    relacoes_grafo: list[RelacaoGrafo]


class ExtracaoGrafoTI(BaseModel):
    metadados_documento: MetadadosDocumento
    conteudo_estruturado: list[SecaoConteudo]

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
    "Divida o conteúdo em seções lógicas (capítulos/tópicos). Para cada seção, extraia um resumo narrativo denso, "
    "os conceitos-chave (com vantagens, desvantagens e casos de uso), tabelas comparativas presentes, "
    "figuras/diagramas relevantes (descrevendo o que mostram passo a passo) e as relações de grafo entre as entidades "
    "identificadas (dependência, implementação, mitigação de risco, substituição, comunicação etc.). "
    "Preencha também os metadados gerais do documento. Gere a resposta seguindo estritamente o schema JSON fornecido."
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
        max_tokens=MAX_TOKENS_SAIDA,
    )

    escolha = response.choices[0]
    if getattr(escolha, "finish_reason", None) == "length":
        print(
            f"        ⚠️  [Aviso] Resposta foi CORTADA por limite de tokens "
            f"(max_tokens={MAX_TOKENS_SAIDA}). O JSON provavelmente virá truncado."
        )

    print("        [Controle de Taxa] Pausando por 20 segundos para preservar os créditos...")
    time.sleep(20)
    return escolha.message.content


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
        max_tokens=MAX_TOKENS_SAIDA,
        extra_body={"provider": {"only": ["google-ai-studio"]}},  # exige provedor com suporte a link do YouTube
    )

    escolha = response.choices[0]
    if getattr(escolha, "finish_reason", None) == "length":
        print(
            f"        ⚠️  [Aviso] Resposta foi CORTADA por limite de tokens "
            f"(max_tokens={MAX_TOKENS_SAIDA}). O JSON provavelmente virá truncado."
        )

    print("        [Controle de Taxa] Pausando por 20 segundos para preservar os créditos...")
    time.sleep(20)
    return escolha.message.content


def salvar_dados_para_graphrag(json_str: str, nome_base: str, pasta_saida: str = "./input"):
    """Achata o JSON estruturado (metadados + seções + conceitos + tabelas +
    figuras + relações) em texto corrido denso, que é o formato que o
    GraphRAG realmente consome (input: type: text). O GraphRAG não entende
    JSON/tabelas/objetos — ele só chunkeia texto puro e extrai entidades
    dele, então quanto mais explícito e narrativo o texto, melhor o grafo
    final do GraphRAG fica."""
    os.makedirs(pasta_saida, exist_ok=True)

    try:
        dados = json.loads(json_str)
    except json.JSONDecodeError as erro_original:
        # Fallback: tenta reparar JSON malformado (aspas não escapadas,
        # string truncada por corte de tokens, vírgula sobrando, etc.)
        # antes de desistir. Cobre os casos que o max_tokens mais alto
        # não resolver sozinho.
        print(
            f"        ⚠️  [Aviso] JSON malformado ({erro_original}). "
            f"Tentando reparo automático com json_repair..."
        )
        try:
            json_str_reparado = repair_json(json_str)
            dados = json.loads(json_str_reparado)
            print("        ✅ [Reparo] JSON reparado com sucesso.")
        except Exception:
            # Salva o JSON bruto para inspeção manual em vez de só perder o conteúdo.
            os.makedirs("./json_com_falha", exist_ok=True)
            caminho_bruto = os.path.join("./json_com_falha", f"{nome_base}_{int(time.time())}.json")
            with open(caminho_bruto, "w", encoding="utf-8") as f_bruto:
                f_bruto.write(json_str)
            print(f"        📄 [Diagnóstico] JSON bruto salvo em: {caminho_bruto}")
            raise

    timestamp = int(time.time())
    nome_arquivo = f"conhecimento_ti_{nome_base}_{timestamp}.txt"
    caminho_final = os.path.join(pasta_saida, nome_arquivo)

    meta = dados.get("metadados_documento", {})
    secoes = dados.get("conteudo_estruturado", [])

    with open(caminho_final, "w", encoding="utf-8") as f:
        # --- Metadados do documento ---
        f.write("=== METADADOS DO DOCUMENTO ===\n")
        f.write(f"Titulo: {meta.get('titulo', '')}\n")
        f.write(f"Idioma: {meta.get('idioma', '')}\n")
        tecnologias = meta.get("tecnologias_citadas", [])
        if tecnologias:
            f.write(f"Tecnologias citadas no documento: {', '.join(tecnologias)}\n")
        f.write("\n")

        for secao in secoes:
            titulo_secao = secao.get("titulo_secao", "")
            secao_id = secao.get("secao_id", "")
            f.write(f"=== SECAO [{secao_id}]: {titulo_secao} ===\n")

            texto_base = secao.get("texto_base", {})
            resumo = texto_base.get("resumo_narrativo", "")
            if resumo:
                f.write(f"{resumo}\n\n")

            # --- Conceitos-chave ---
            conceitos = texto_base.get("conceitos_chave", [])
            if conceitos:
                f.write("-- Conceitos-chave --\n")
                for c in conceitos:
                    f.write(
                        f"Termo: {c.get('termo')} | Tipo: {c.get('tipo_entidade')} | "
                        f"Definicao: {c.get('definicao')}\n"
                    )
                    atrib = c.get("atributos_tecnicos", {})
                    vantagens = atrib.get("vantagens", [])
                    desvantagens = atrib.get("desvantagens", [])
                    casos_de_uso = atrib.get("casos_de_uso", [])
                    if vantagens:
                        f.write(f"  Vantagens de {c.get('termo')}: {'; '.join(vantagens)}\n")
                    if desvantagens:
                        f.write(f"  Desvantagens de {c.get('termo')}: {'; '.join(desvantagens)}\n")
                    if casos_de_uso:
                        f.write(f"  Casos de uso de {c.get('termo')}: {'; '.join(casos_de_uso)}\n")
                f.write("\n")

            # --- Tabelas (viram texto narrativo linha a linha) ---
            tabelas = secao.get("tabelas", [])
            if tabelas:
                f.write("-- Tabelas comparativas --\n")
                for t in tabelas:
                    f.write(f"Tabela '{t.get('titulo_tabela')}': {t.get('descricao_contexto', '')}\n")
                    for linha in t.get("dados_linhas", []):
                        pares = "; ".join(
                            f"{v.get('coluna')} = {v.get('valor')}" for v in linha.get("valores", [])
                        )
                        f.write(f"  Linha {linha.get('id_linha')}: {pares}\n")
                        relacionadas = linha.get("entidades_relacionadas", [])
                        if relacionadas:
                            f.write(f"    (Relacionado a: {', '.join(relacionadas)})\n")
                f.write("\n")

            # --- Figuras e gráficos ---
            figuras = secao.get("figuras_e_graficos", [])
            if figuras:
                f.write("-- Figuras e diagramas --\n")
                for fig in figuras:
                    f.write(
                        f"Figura '{fig.get('titulo_figura')}' ({fig.get('tipo_visual')}): "
                        f"{fig.get('dados_extraidos', '')}\n"
                    )
                    componentes = fig.get("componentes_visuais", [])
                    if componentes:
                        f.write(f"  Componentes visuais: {', '.join(componentes)}\n")
                    relacionadas = fig.get("entidades_relacionadas", [])
                    if relacionadas:
                        f.write(f"  Relacionado a: {', '.join(relacionadas)}\n")
                f.write("\n")

            # --- Relações do grafo ---
            relacoes = secao.get("relacoes_grafo", [])
            if relacoes:
                f.write("-- Relacionamentos --\n")
                for rel in relacoes:
                    f.write(
                        f"Conexao: '{rel.get('origem')}' --[{rel.get('tipo_relacao')}]--> "
                        f"'{rel.get('destino')}'. Motivo: {rel.get('descricao_conexao', '')}\n"
                    )
                f.write("\n")

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