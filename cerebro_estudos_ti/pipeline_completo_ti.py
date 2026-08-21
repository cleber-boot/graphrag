import os
import time
import json
import glob
import shutil
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# =====================================================================
# 1. CONFIGURAÇÃO INICIAL E CONTRATO DE DADOS DE TI
# =====================================================================

client = genai.Client()

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
    "Analise minuciosamente o material anexado (seja texto, vídeo ou áudio de aula). Identifique todas as tecnologias, "
    "protocolos, arquiteturas, algoritmos, pegadinhas de bancas e questões presentes. Gere uma lista estrita de Nós e Relacionamentos em JSON."
)

def obter_config_extracao():
    return types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=ExtracaoGrafoTI,
        temperature=0.1,
    )

# =====================================================================
# 2. MOTORES DE PROCESSAMENTO MULTIMÍDIA E WEB (GEMINI CHATS)
# =====================================================================

def processar_link_youtube(url: str, descricao_materia: str):
    print(f"\n[Fase 1] Analisando Link do YouTube de TI: {url}")
    part = types.Part.from_uri(file_uri=url, mime_type="video/mp4")
    
    chat = client.chats.create(
        model='gemini-2.5-flash-lite',
        config=obter_config_extracao()
    )
    response = chat.send_message(
        message=[part, f"{PROMPT_COMANDO}\nContexto do Material: {descricao_materia}"]
    )
    
    print("        [Controle de Taxa] Pausando por 20 segundos para preservar a cota da API...")
    time.sleep(20)
    return response.text

def processar_arquivo_local(caminho_arquivo: str, mime_type: str, descricao_materia: str):
    print(f"\n[Fase 1] Fazendo upload do arquivo para o ecossistema Google Cloud: {caminho_arquivo}")
    arquivo_remoto = client.files.upload(file=caminho_arquivo)
    
    while arquivo_remoto.state.name == "PROCESSING":
        print("        Aguardando indexação multimídia do arquivo nos servidores da Google (7 segundos)...")
        time.sleep(7)
        arquivo_remoto = client.files.get(name=arquivo_remoto.name)
        
    if arquivo_remoto.state.name != "ACTIVE":
        raise Exception(f"Falha na indexação do arquivo: {arquivo_remoto.state.name}")
        
    print(f"        Arquivo ativo e pronto! Extraindo grafo relacional de TI via Chat Session...")
    try:
        chat = client.chats.create(
            model='gemini-2.5-flash-lite',
            config=obter_config_extracao()
        )
        response = chat.send_message(
            message=[arquivo_remoto, f"{PROMPT_COMANDO}\nContexto do Material: {descricao_materia}"]
        )
        dados_json = response.text
    finally:
        print("        Limpando e deletando arquivo temporário do servidor da Google...")
        client.files.delete(name=arquivo_remoto.name)
        
    print("        [Controle de Taxa] Pausando por 20 segundos para preservar a cota da API...")
    time.sleep(20)
    return dados_json

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
                if "429" in str(e) or "quota" in str(e).lower(): return False
    return True

def processar_links_txt_automatico():
    arquivo_links = "links.txt"
    if not os.path.exists(arquivo_links):
        print("\n[Aviso] Arquivo 'links.txt' não encontrado na raiz. Pulando leitura de links.")
        return True
        
    with open(arquivo_links, "r", encoding="utf-8") as f:
        links = [linha.strip() for list_linha in f if (linha := list_linha.strip()) and not linha.startswith("#")]
        
    if not links:
        print("\n[Links] O arquivo 'links.txt' está vazio ou sem links válidos.")
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
            if "429" in str(e) or "quota" in str(e).lower():
                print("🛑 [Cota Excedida] Interrompendo lote de links para proteção.")
                break
                
    # Atualiza o arquivo de links mantendo apenas o que NÃO foi processado por falta de cota
    links_restantes = [l for l in links if l not in links_processados_com_sucesso]
    with open(arquivo_links, "w", encoding="utf-8") as f:
        for l in links_restantes:
            f.write(f"{l}\n")
            
    if not links_restantes:
        print("🧹 [Limpeza] Todos os links foram processados! O arquivo 'links.txt' foi limpo.")
    else:
        print(f"⚠️ [Aviso] {len(links_restantes)} links falharam ou pararam na cota e ficaram salvos para a próxima execução.")

if __name__ == "__main__":
    if not os.environ.get("GEMINI_API_KEY"):
        print("ERRO: Defina a variável de ambiente GEMINI_API_KEY."); exit(1)
        
    print("=== INICIANDO ECOSSISTEMA TOTAL DE TI (ARQUIVOS + LINKS) ===")
    
    # Executa primeiro os arquivos locais e depois a lista de links
    sucesso_locais = processar_arquivos_locais_automatico()
    if sucesso_locais:
        processar_links_txt_automatico()
        
    print("\n=== PIPELINE DE EXTRAÇÃO CONCLUÍDO ===")
