"""
Chat com memória persistente para o GraphRAG (cerebro_estudos_ti)
-------------------------------------------------------------------
Interface em Streamlit que roda `graphrag query` por baixo dos panos
e SALVA cada conversa em disco, permitindo:
  - Continuar uma conversa depois de fechar e reabrir o app
  - Navegar pelo histórico de conversas anteriores (tipo um "caderno de estudos")

As conversas ficam salvas em ./chat_sessions/*.json (dentro da raiz
do projeto graphrag), como arquivos de texto simples.

Uso:
    streamlit run chat_app.py
"""

import json
import re
import subprocess
import uuid
import os
from datetime import datetime
from pathlib import Path

import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

import simulado_graph as sg

# set_page_config precisa ser sempre o primeiro comando Streamlit do script.
st.set_page_config(page_title="Chat GraphRAG", page_icon="🧠", layout="centered")

# Resolve os caminhos com base em ONDE ESTE ARQUIVO ESTÁ SALVO, não na pasta
# de onde o comando `streamlit run` foi disparado. Isso evita quebrar quando
# o app é executado a partir de um diretório diferente do projeto.
SCRIPT_DIR = Path(__file__).resolve().parent
load_dotenv(SCRIPT_DIR / ".env")

# ---------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------
ROOT_DIR = str(SCRIPT_DIR)
SESSIONS_DIR = Path(ROOT_DIR) / "chat_sessions"
SESSIONS_DIR.mkdir(exist_ok=True)

MAX_HISTORY_TURNS = 4
TIMEOUT_SECONDS = 300

# Cliente direto para a OpenRouter, usado SÓ na etapa de geração do simulado
# (a busca de contexto continua passando pelo GraphRAG via CLI).
_api_key = os.environ.get("GRAPHRAG_API_KEY")
if not _api_key:
    st.error(
        f"⚠️ Não encontrei a variável GRAPHRAG_API_KEY. Verifique se existe um "
        f"arquivo `.env` com essa chave em: `{SCRIPT_DIR / '.env'}`"
    )
    st.stop()

openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=_api_key,
)
MODELO_SIMULADO = "google/gemini-2.5-flash-lite"

METHOD_INFO = {
    "local": "Perguntas específicas sobre um conceito, entidade ou tópico pontual.",
    "global": "Perguntas amplas sobre o conjunto todo dos documentos (temas gerais).",
    "drift": "Meio-termo entre local e global — mais detalhado, porém mais lento/caro.",
}

# ---------------------------------------------------------------
# Prompt para geração de simulados no estilo de banca de concurso
# ---------------------------------------------------------------
PROMPT_SIMULADO = """Você é um elaborador de provas experiente, especializado em reproduzir fielmente o estilo \
da banca {banca} em concursos públicos de Tecnologia da Informação.

MATERIAL DE REFERÊNCIA (extraído da base de conhecimento do candidato):
\"\"\"
{contexto}
\"\"\"

Com base EXCLUSIVAMENTE no material de referência acima, crie um simulado com {quantidade} questões \
{sobre_tema}seguindo rigorosamente as características da banca {banca}:
- Reproduza o nível de dificuldade, a forma de redigir o enunciado e o estilo de pegadinha típicos dessa banca.
- Use apenas conceitos, tecnologias e relações que estejam de fato presentes no material de referência acima.
- Não invente informações que não estejam no material.

FORMATAÇÃO OBRIGATÓRIA (siga exatamente esta estrutura em Markdown, sem exceções):

## Questão 1

(enunciado da questão em um parágrafo)

{formato_alternativas}

**Gabarito:** (letra ou CERTO/ERRADO)

> **Comentário:** (explicação de por que a alternativa correta está certa e por que as demais estão erradas,
> destacando pegadinhas típicas da banca {banca})

---

Repita exatamente esse padrão (título "## Questão N", linha em branco, enunciado, linha em branco, alternativas \
cada uma em sua própria linha, linha em branco, Gabarito em negrito, linha em branco, Comentário como citação \
em bloco, separador "---") para cada uma das {quantidade} questões, numerando de 1 a {quantidade}.

Nunca escreva o enunciado, as alternativas, o gabarito e o comentário em um único parágrafo corrido — sempre \
separados por linhas em branco como mostrado acima. Responda em português do Brasil.
"""

FORMATOS_QUESTAO = {
    "Múltipla escolha (A–E)": (
        "As cinco alternativas, cada uma em sua PRÓPRIA linha, no formato:\n"
        "**A)** texto da alternativa\n**B)** texto da alternativa\n**C)** texto da alternativa\n"
        "**D)** texto da alternativa\n**E)** texto da alternativa\n(apenas uma correta)"
    ),
    "Certo ou Errado (CESPE-like)": "Uma única afirmação, a ser julgada como CERTO ou ERRADO.",
}

FORMATO_JSON_DESC = {
    "Múltipla escolha (A–E)": "cinco alternativas (A, B, C, D, E), apenas uma correta",
    "Certo ou Errado (CESPE-like)": "uma única afirmação a ser julgada como CERTO ou ERRADO",
}

PERGUNTA_BUSCA_CONTEXTO = (
    "Quais são os principais conceitos, tecnologias, protocolos, arquiteturas e relações técnicas {sobre_tema}? "
    "Traga definições, características técnicas, comparações e detalhes relevantes com o máximo de profundidade "
    "possível, incluindo nomes específicos de normas, padrões e siglas."
)


# ---------------------------------------------------------------
# Persistência em disco
# ---------------------------------------------------------------
def list_sessions() -> list[dict]:
    """Retorna metadados de todas as conversas salvas, mais recentes primeiro."""
    sessions = []
    for f in SESSIONS_DIR.glob("*.json"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            sessions.append({
                "id": f.stem,
                "title": data.get("title", f.stem),
                "updated_at": data.get("updated_at", ""),
                "path": f,
            })
        except (json.JSONDecodeError, OSError):
            continue
    sessions.sort(key=lambda s: s["updated_at"], reverse=True)
    return sessions


def load_session(session_id: str) -> dict:
    path = SESSIONS_DIR / f"{session_id}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"title": "Nova conversa", "created_at": now_iso(), "updated_at": now_iso(), "messages": []}


def save_session(session_id: str, title: str, messages: list[dict]) -> None:
    path = SESSIONS_DIR / f"{session_id}.json"
    existing_created_at = now_iso()
    if path.exists():
        try:
            existing_created_at = json.loads(path.read_text(encoding="utf-8")).get("created_at", now_iso())
        except (json.JSONDecodeError, OSError):
            pass
    data = {
        "title": title,
        "created_at": existing_created_at,
        "updated_at": now_iso(),
        "messages": messages,
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def build_texto_chat(titulo: str, mensagens: list[dict]) -> str:
    """Monta uma versão em texto simples de toda a conversa, pronta para copiar
    (usa o mesmo Markdown do simulado quando a mensagem for um simulado)."""
    linhas = [f"# {titulo}", ""]
    for msg in mensagens:
        rotulo = "Você" if msg["role"] == "user" else "Assistente"
        linhas.append(f"### {rotulo}")
        if eh_simulado_estruturado(msg):
            linhas.append(sg.para_markdown(msg["dados"]))
        else:
            linhas.append(str(msg.get("content", "")))
        linhas.append("")
    return "\n".join(linhas).strip()


def _chave_js(bruta: str) -> str:
    """Sanitiza uma key qualquer para servir de nome de variável/id JS válido."""
    return re.sub(r"[^0-9a-zA-Z_]", "_", bruta)


def render_acoes_mensagem(texto: str, key: str) -> None:
    """Desenha dois botõezinhos abaixo de UMA mensagem: copiar (clipboard do
    navegador) e imprimir (abre a mensagem sozinha numa aba nova e chama
    window.print(), imprimindo só aquele conteúdo, não o app inteiro)."""
    k = _chave_js(key)
    texto_js = json.dumps(texto)  # string JS já escapada (aspas, quebras de linha etc.)

    html = f"""
    <div style="display:flex; gap:6px; margin:2px 0 10px 0; font-family:-apple-system,sans-serif;">
      <button id="copiar_{k}" style="
          font-size:12px; padding:4px 10px; border-radius:6px;
          border:1px solid #4a4a4a55; background:transparent; color:inherit; cursor:pointer;">
        📋 Copiar
      </button>
      <button id="imprimir_{k}" style="
          font-size:12px; padding:4px 10px; border-radius:6px;
          border:1px solid #4a4a4a55; background:transparent; color:inherit; cursor:pointer;">
        🖨️ Imprimir
      </button>
    </div>
    <script>
      (function() {{
        const texto = {texto_js};

        const btnCopiar = document.getElementById("copiar_{k}");
        btnCopiar.addEventListener("click", function() {{
          navigator.clipboard.writeText(texto).then(function() {{
            btnCopiar.innerText = "✅ Copiado!";
            setTimeout(function() {{ btnCopiar.innerText = "📋 Copiar"; }}, 1500);
          }}).catch(function() {{
            btnCopiar.innerText = "⚠️ Falhou";
            setTimeout(function() {{ btnCopiar.innerText = "📋 Copiar"; }}, 1500);
          }});
        }});

        const btnImprimir = document.getElementById("imprimir_{k}");
        btnImprimir.addEventListener("click", function() {{
          const esc = texto
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;");
          const janela = window.open("", "_blank");
          if (!janela) {{ return; }}
          janela.document.write(
            "<html><head><title>Imprimir</title></head><body>" +
            "<pre style=\\"white-space:pre-wrap;font-family:sans-serif;font-size:14px;line-height:1.5;padding:24px;\\">" +
            esc + "</pre></body></html>"
          );
          janela.document.close();
          janela.focus();
          setTimeout(function() {{ janela.print(); }}, 200);
        }});
      }})();
    </script>
    """
    st.iframe(html, height="content")


def now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def new_session_id() -> str:
    return uuid.uuid4().hex[:8]


# ---------------------------------------------------------------
# Estado da sessão (Streamlit)
# ---------------------------------------------------------------
if "session_id" not in st.session_state:
    # Ao abrir o app, carrega a conversa mais recente (se existir) ou cria uma nova.
    existing = list_sessions()
    if existing:
        st.session_state.session_id = existing[0]["id"]
        loaded = load_session(existing[0]["id"])
        st.session_state.messages = loaded["messages"]
        st.session_state.title = loaded["title"]
    else:
        st.session_state.session_id = new_session_id()
        st.session_state.messages = []
        st.session_state.title = "Nova conversa"

st.title(f"🧠 {st.session_state.title}")

# ---------------------------------------------------------------
# Controle global: expandir/colapsar todos os gabaritos de uma vez
# ---------------------------------------------------------------
def eh_mensagem_simulado(msg: dict) -> bool:
    """Detecta se uma mensagem é um simulado, mesmo em conversas salvas
    ANTES da marcação 'kind' existir (compatibilidade com histórico antigo)."""
    if msg.get("kind") in ("simulado", "simulado_json"):
        return True
    return msg.get("role") == "assistant" and "**Gabarito:**" in msg.get("content", "")


def eh_simulado_estruturado(msg: dict) -> bool:
    return msg.get("kind") == "simulado_json" and isinstance(msg.get("dados"), dict)


for _flag in ("mostrar_gabaritos", "mostrar_base", "mostrar_micro"):
    if _flag not in st.session_state:
        st.session_state[_flag] = False

tem_simulado_na_conversa = any(eh_mensagem_simulado(m) for m in st.session_state.messages)
tem_estruturado = any(eh_simulado_estruturado(m) for m in st.session_state.messages)

if tem_simulado_na_conversa:
    col_g, col_b, col_m = st.columns(3)
    with col_g:
        rotulo = "🙈 Colapsar gabaritos" if st.session_state.mostrar_gabaritos else "👁️ Expandir gabaritos"
        if st.button(rotulo, use_container_width=True):
            st.session_state.mostrar_gabaritos = not st.session_state.mostrar_gabaritos
            st.rerun()
    if tem_estruturado:
        with col_b:
            rotulo = "🙈 Colapsar base" if st.session_state.mostrar_base else "📚 Expandir base de conhecimento"
            if st.button(rotulo, use_container_width=True):
                st.session_state.mostrar_base = not st.session_state.mostrar_base
                st.rerun()
        with col_m:
            rotulo = "🙈 Colapsar micro questões" if st.session_state.mostrar_micro else "🧩 Expandir micro questões"
            if st.button(rotulo, use_container_width=True):
                st.session_state.mostrar_micro = not st.session_state.mostrar_micro
                st.rerun()


# ---------------------------------------------------------------
# Barra lateral
# ---------------------------------------------------------------
with st.sidebar:
    st.header("Configurações")
    method = st.selectbox(
        "Método de busca", options=["local", "global", "drift"], index=0,
        help="Escolha o tipo de busca do GraphRAG.",
    )
    st.caption(METHOD_INFO[method])

    community_level = st.slider(
        "Nível de comunidade", min_value=0, max_value=4, value=2,
        help="Controla a granularidade das comunidades usadas na busca (quando aplicável).",
    )

    use_memory = st.checkbox("Manter memória dentro da conversa", value=True)

    st.divider()
    st.subheader("📝 Gerar simulado")

    banca = st.text_input("Banca", value="FGV", help="Ex: FGV, CESPE/CEBRASPE, FCC, VUNESP...")
    formato_questao = st.selectbox("Formato da questão", options=list(FORMATOS_QUESTAO.keys()))

    origem = st.radio(
        "Origem das questões",
        options=["Conhecimentos do grafo", "Tema livre"],
        help="No modo grafo você escolhe quantas questões quer de cada conhecimento indexado.",
    )

    # Estas variáveis são preenchidas por um dos dois modos abaixo.
    tema_simulado = ""
    quantidade_questoes = 0
    plano_conhecimentos: list[dict] = []

    if origem == "Conhecimentos do grafo":
        assinatura = sg.assinatura_indice(ROOT_DIR)
        conhecimentos = sg.carregar_conhecimentos(ROOT_DIR, assinatura)

        if conhecimentos.empty:
            st.warning(
                "Não encontrei `output/community_reports.parquet`. "
                "Rode a indexação do GraphRAG antes de usar este modo."
            )
        else:
            niveis_disponiveis = sorted(conhecimentos["level"].unique().tolist())
            niveis = st.multiselect(
                "Granularidade (nível das comunidades)",
                options=niveis_disponiveis,
                default=[niveis_disponiveis[0]],
                help="Nível 0 = conhecimentos mais amplos. Níveis maiores = temas mais específicos.",
            )
            busca_conhecimento = st.text_input(
                "Filtrar conhecimentos", placeholder="Ex: cabeamento, VLAN, Wi-Fi..."
            )

            filtrados = sg.filtrar_conhecimentos(conhecimentos, niveis, busca_conhecimento)
            st.caption(f"{len(filtrados)} conhecimento(s) disponível(is).")

            selecionados = st.multiselect(
                "Conhecimentos do grafo",
                options=filtrados["rotulo"].tolist(),
                help="Selecione um ou mais. Para cada um você define a quantidade de questões.",
            )

            if selecionados:
                st.caption("Quantidade de questões por conhecimento:")
                por_rotulo = filtrados.set_index("rotulo")
                for rotulo in selecionados:
                    linha = por_rotulo.loc[rotulo]
                    qtd = st.number_input(
                        linha["title"][:45],
                        min_value=1,
                        max_value=20,
                        value=3,
                        step=1,
                        key=f"qtd_{linha['community']}_{linha['level']}",
                    )
                    plano_conhecimentos.append({
                        "community": int(linha["community"]),
                        "titulo": str(linha["title"]),
                        "quantidade": int(qtd),
                        "linha": linha,
                    })
                total = sum(p["quantidade"] for p in plano_conhecimentos)
                st.info(f"Total: **{total}** questões em {len(plano_conhecimentos)} conhecimento(s).")
    else:
        tema_simulado = st.text_input(
            "Tema (opcional)", placeholder="Ex: redes de computadores, segurança da informação..."
        )
        quantidade_questoes = st.number_input(
            "Quantidade de questões", min_value=1, max_value=20, value=5, step=1
        )

    st.caption("Apoio ao estudo")
    incluir_base = st.checkbox(
        "📚 Incluir base de conhecimento", value=True,
        help="Blocos de teoria necessários para resolver a questão, exibidos colapsados.",
    )
    incluir_micro = st.checkbox(
        "🧩 Incluir micro questões", value=True,
        help="Questões menores que decompõem o raciocínio da questão principal.",
    )
    n_micro = st.slider("Micro questões por questão", 1, 5, 3, disabled=not incluir_micro)

    gerar_simulado_clicado = st.button("🎯 Gerar simulado agora", use_container_width=True, type="primary")

    st.divider()
    st.subheader("📋 Copiar conversa")

    if "mostrar_copia_chat" not in st.session_state:
        st.session_state.mostrar_copia_chat = False

    if st.button("📋 Copiar chat inteiro", use_container_width=True):
        st.session_state.mostrar_copia_chat = not st.session_state.mostrar_copia_chat

    if st.session_state.mostrar_copia_chat:
        if not st.session_state.messages:
            st.caption("Esta conversa ainda não tem mensagens.")
        else:
            texto_chat = build_texto_chat(st.session_state.title, st.session_state.messages)
            st.caption("Clique no ícone 📄 no canto do bloco abaixo para copiar tudo.")
            st.code(texto_chat, language=None, wrap_lines=True)

    st.divider()
    st.subheader("💬 Conversas salvas")

    if st.button("➕ Nova conversa", use_container_width=True):
        st.session_state.session_id = new_session_id()
        st.session_state.messages = []
        st.session_state.title = "Nova conversa"
        st.rerun()

    sessions = list_sessions()
    if sessions:
        for s in sessions:
            is_current = s["id"] == st.session_state.session_id
            label = f"{'👉 ' if is_current else ''}{s['title']} · {s['updated_at'][:16]}"
            if st.button(label, key=f"load_{s['id']}", use_container_width=True):
                loaded = load_session(s["id"])
                st.session_state.session_id = s["id"]
                st.session_state.messages = loaded["messages"]
                st.session_state.title = loaded["title"]
                st.rerun()
    else:
        st.caption("Nenhuma conversa salva ainda.")

    st.divider()
    new_title = st.text_input("Renomear conversa atual", value=st.session_state.title)
    if new_title != st.session_state.title:
        st.session_state.title = new_title
        save_session(st.session_state.session_id, st.session_state.title, st.session_state.messages)

    if st.button("🗑️ Apagar esta conversa", use_container_width=True):
        path = SESSIONS_DIR / f"{st.session_state.session_id}.json"
        if path.exists():
            path.unlink()
        st.session_state.session_id = new_session_id()
        st.session_state.messages = []
        st.session_state.title = "Nova conversa"
        st.rerun()

    st.divider()
    st.caption(f"Raiz do projeto: `{ROOT_DIR}`")
    st.caption(f"Conversas salvas em: `{SESSIONS_DIR}`")


# ---------------------------------------------------------------
# Funções de consulta ao GraphRAG
# ---------------------------------------------------------------
def build_query_with_context(new_question: str) -> str:
    if not use_memory or not st.session_state.messages:
        return new_question

    history = st.session_state.messages[-(MAX_HISTORY_TURNS * 2):]
    context_lines = ["Histórico da conversa até agora:"]
    for msg in history:
        prefix = "Usuário" if msg["role"] == "user" else "Assistente"
        context_lines.append(f"{prefix}: {msg['content']}")
    context_lines.append("")
    context_lines.append(f"Nova pergunta do usuário: {new_question}")
    context_lines.append(
        "\nResponda a nova pergunta levando em conta o histórico acima quando fizer sentido."
    )
    return "\n".join(context_lines)


def clean_output(raw: str) -> str:
    lines = raw.splitlines()
    cleaned = []
    skip_patterns = [
        r"^INFO[:\s]", r"^DEBUG[:\s]", r"^WARNING[:\s]",
        r"^\s*$",
        r"reading table from storage",
        r"Vector Store Args",
    ]
    for line in lines:
        if any(re.search(p, line) for p in skip_patterns):
            continue
        cleaned.append(line)
    return "\n".join(cleaned).strip()


def build_simulado_prompt(banca: str, tema: str, quantidade: int, formato_label: str, contexto: str) -> str:
    sobre_tema = f"sobre o tema '{tema}' " if tema.strip() else ""
    return PROMPT_SIMULADO.format(
        banca=banca.strip() or "FGV",
        quantidade=quantidade,
        sobre_tema=sobre_tema,
        formato_alternativas=FORMATOS_QUESTAO[formato_label],
        contexto=contexto,
    )


def buscar_contexto_para_simulado(tema: str, method: str, community_level: int) -> str:
    """Etapa 1: pergunta simples e natural ao GraphRAG, que ele sabe responder bem
    (evita o prompt gigante de formatação confundir a busca)."""
    sobre_tema = f"sobre o tema '{tema}'" if tema.strip() else "presentes no material"
    pergunta = PERGUNTA_BUSCA_CONTEXTO.format(sobre_tema=sobre_tema)
    return run_graphrag_query(pergunta, method, community_level)


def gerar_simulado_via_llm(contexto: str, banca: str, tema: str, quantidade: int, formato_label: str) -> str:
    """Etapa 2: chamada DIRETA à OpenRouter (fora do GraphRAG) para elaborar
    as questões formatadas, usando o contexto já buscado na etapa 1."""
    prompt = build_simulado_prompt(banca, tema, quantidade, formato_label, contexto)

    response = openrouter_client.chat.completions.create(
        model=MODELO_SIMULADO,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response.choices[0].message.content


def formatar_simulado(texto: str) -> str:
    """Reformata em Markdown de forma consistente. Em vez de tentar 'detectar'
    o que já está formatado (abordagem anterior, que falhava quando o modelo
    colocava negrito sem quebra de linha real), esta versão LIMPA toda
    formatação prévia primeiro e depois reconstrói do zero, garantindo que
    cada alternativa sempre comece em uma linha nova, na margem esquerda."""

    # 1) Remove formatação prévia (negrito, cabeçalhos, citação em bloco)
    #    para não depender de o modelo ter formatado certo ou não.
    texto = re.sub(r"\*\*", "", texto)
    texto = re.sub(r"^#+\s*", "", texto, flags=re.MULTILINE)
    texto = re.sub(r"^>\s*", "", texto, flags=re.MULTILINE)

    # 2) Quebra antes de cada "Questão N" (aceita maiúsc./minúsc.)
    texto = re.sub(
        r"\s*[Qq]uest(?:ã|a)o\s+(\d+)\b\s*",
        r"\n\n---\n\n## Questão \1\n\n",
        texto,
    )
    # 3) Quebra SEMPRE antes de cada alternativa A) a E), garantindo que cada
    #    uma comece em uma linha nova, na margem esquerda (sem indentação).
    texto = re.sub(r"\s*\b([A-E])\)\s+", r"\n\n\1) ", texto)
    # Negrito só na letra + parêntese, no início da linha
    texto = re.sub(r"^([A-E])\) ", r"**\1)** ", texto, flags=re.MULTILINE)
    # 4) Destaca o gabarito em negrito, em linha própria
    texto = re.sub(r"\s*[Gg][Aa][Bb][Aa][Rr][Ii][Tt][Oo]:\s*", r"\n\n**Gabarito:** ", texto)
    # 5) Formata o comentário como citação em bloco
    texto = re.sub(r"\s*[Cc]oment[áa]rio:\s*", r"\n\n> **Comentário:** ", texto)

    return texto.strip()


def render_simulado(texto: str, msg_id) -> None:
    """Renderiza o simulado questão por questão, escondendo o gabarito e o
    comentário dentro de um expansor. O estado inicial (aberto/fechado) de
    TODOS os expansores é controlado por st.session_state.mostrar_gabaritos.

    Cada expansor recebe uma `key` que inclui o estado atual do botão mestre
    (mostrar_gabaritos). Isso força o Streamlit a tratá-lo como um componente
    NOVO sempre que o botão mestre é clicado, ignorando qualquer clique manual
    anterior do usuário em um expansor individual — garantindo que todos
    fiquem realmente sincronizados de uma vez."""
    expandido = st.session_state.get("mostrar_gabaritos", False)

    blocos = re.split(r"\n\n---\n\n", texto)
    q_idx = 0
    for bloco in blocos:
        bloco = bloco.strip()
        if not bloco:
            continue

        match = re.search(r"\*\*Gabarito:\*\*", bloco)
        if match:
            parte_questao = bloco[: match.start()].strip()
            parte_questao = re.sub(r"^-+\s*", "", parte_questao)  # remove '---' solto no início
            parte_gabarito = bloco[match.start():].strip()
            st.markdown(parte_questao)
            chave = f"gab_{msg_id}_{q_idx}_{expandido}"
            with st.expander("👁️ Ver gabarito e comentário", expanded=expandido, key=chave):
                st.markdown(parte_gabarito)
            q_idx += 1
        else:
            bloco = re.sub(r"^-+\s*", "", bloco)
            st.markdown(bloco)


def run_graphrag_query(question: str, method: str, community_level: int) -> str:
    cmd = [
        "python", "-m", "graphrag", "query",
        "--root", ROOT_DIR,
        "--method", method,
        "--community-level", str(community_level),
        question,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        return "⚠️ A consulta demorou demais e foi cancelada (timeout). Tente novamente ou simplifique a pergunta."

    if result.returncode != 0:
        error_msg = result.stderr.strip() or result.stdout.strip()
        return f"❌ Erro ao consultar o GraphRAG:\n\n```\n{error_msg[-2000:]}\n```"

    return clean_output(result.stdout) or "(Sem resposta retornada pelo GraphRAG.)"


# ---------------------------------------------------------------
# Renderiza histórico
# ---------------------------------------------------------------
for _idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        if eh_simulado_estruturado(msg):
            sg.render_simulado_estruturado(
                msg["dados"],
                prefixo=f"hist_{_idx}",
                mostrar_base=st.session_state.mostrar_base,
                mostrar_micro=st.session_state.mostrar_micro,
                mostrar_gabarito=st.session_state.mostrar_gabaritos,
            )
            render_acoes_mensagem(sg.para_markdown(msg["dados"]), key=f"hist_{_idx}")
        elif eh_mensagem_simulado(msg):
            render_simulado(msg["content"], msg_id=f"hist_{_idx}")
            render_acoes_mensagem(msg["content"], key=f"hist_{_idx}")
        else:
            st.markdown(msg["content"])
            render_acoes_mensagem(msg["content"], key=f"hist_{_idx}")

# ---------------------------------------------------------------
# Geração de simulado (via botão na barra lateral)
# ---------------------------------------------------------------
if gerar_simulado_clicado:
    modo_grafo = origem == "Conhecimentos do grafo"
    certo_errado = formato_questao.startswith("Certo")
    formato_desc = FORMATO_JSON_DESC[formato_questao]

    if modo_grafo and not plano_conhecimentos:
        st.warning("Selecione pelo menos um conhecimento do grafo antes de gerar o simulado.")
        st.stop()

    if modo_grafo:
        total = sum(p["quantidade"] for p in plano_conhecimentos)
        detalhe = " · ".join(f"{p['titulo'][:30]} ({p['quantidade']})" for p in plano_conhecimentos)
        pedido_visivel = f"🎯 Simulado — banca {banca or 'FGV'} · {total} questões\n\n**Conhecimentos:** {detalhe}"
        titulo_auto = f"Simulado {banca or 'FGV'} — {plano_conhecimentos[0]['titulo']}"
    else:
        tema_label = tema_simulado.strip() or "conteúdo geral da base"
        pedido_visivel = f"🎯 Gerar simulado — banca {banca or 'FGV'} · {quantidade_questoes} questões · {tema_label}"
        titulo_auto = f"Simulado {banca or 'FGV'} — {tema_label}"

    st.session_state.messages.append({"role": "user", "content": pedido_visivel})
    with st.chat_message("user"):
        st.markdown(pedido_visivel)

    if len(st.session_state.messages) == 1 and st.session_state.title == "Nova conversa":
        st.session_state.title = titulo_auto[:50]

    questoes: list[dict] = []
    erro_msg = ""

    with st.chat_message("assistant"):
        if modo_grafo:
            mapa_entidades = sg.carregar_entidades_por_comunidade(ROOT_DIR, sg.assinatura_indice(ROOT_DIR))
            barra = st.progress(0.0, text="Preparando...")

            for i, plano in enumerate(plano_conhecimentos):
                barra.progress(
                    i / len(plano_conhecimentos),
                    text=f"Elaborando {plano['quantidade']} questão(ões) de '{plano['titulo'][:40]}'...",
                )
                contexto = sg.montar_contexto(plano["linha"], mapa_entidades.get(plano["community"], []))
                try:
                    novas = sg.gerar_questoes(
                        client=openrouter_client,
                        modelo=MODELO_SIMULADO,
                        contexto=contexto,
                        banca=banca,
                        tema=plano["titulo"],
                        quantidade=plano["quantidade"],
                        formato_descricao=formato_desc,
                        certo_errado=certo_errado,
                        n_micro=n_micro,
                        incluir_base=incluir_base,
                        incluir_micro=incluir_micro,
                    )
                except Exception as exc:  # noqa: BLE001 - mostra o erro para o usuário
                    st.warning(f"Falha ao gerar questões de '{plano['titulo']}': {exc}")
                    continue

                for q in novas:
                    q["_conhecimento"] = plano["titulo"]
                questoes.extend(novas)

            barra.progress(1.0, text="Pronto!")
            barra.empty()
        else:
            metodo_para_simulado = "drift" if method == "drift" else "global"
            with st.spinner("Buscando conteúdo relevante na base de conhecimento..."):
                contexto = buscar_contexto_para_simulado(tema_simulado, metodo_para_simulado, community_level)

            if not contexto or "unable to answer" in contexto.lower() or "não foi possível" in contexto.lower():
                erro_msg = (
                    "⚠️ Não encontrei conteúdo suficiente na base de conhecimento para gerar o simulado "
                    f"{'sobre *' + tema_simulado + '*' if tema_simulado.strip() else 'solicitado'}. "
                    "Tente um tema mais amplo ou use o modo 'Conhecimentos do grafo'."
                )
            else:
                with st.spinner("Elaborando as questões com base no material encontrado..."):
                    try:
                        questoes = sg.gerar_questoes(
                            client=openrouter_client,
                            modelo=MODELO_SIMULADO,
                            contexto=contexto,
                            banca=banca,
                            tema=tema_simulado,
                            quantidade=quantidade_questoes,
                            formato_descricao=formato_desc,
                            certo_errado=certo_errado,
                            n_micro=n_micro,
                            incluir_base=incluir_base,
                            incluir_micro=incluir_micro,
                        )
                    except Exception as exc:  # noqa: BLE001
                        erro_msg = f"⚠️ Não consegui montar o simulado: {exc}"

        if not questoes:
            erro_msg = erro_msg or "⚠️ Nenhuma questão foi gerada. Tente outro conhecimento ou reduza a quantidade."
            st.markdown(erro_msg)
        else:
            dados_simulado = {
                "cabecalho": f"**Simulado — banca {banca or 'FGV'}** · {len(questoes)} questões",
                "questoes": questoes,
            }
            sg.render_simulado_estruturado(
                dados_simulado,
                prefixo=f"live_{len(st.session_state.messages)}",
                mostrar_base=st.session_state.mostrar_base,
                mostrar_micro=st.session_state.mostrar_micro,
                mostrar_gabarito=st.session_state.mostrar_gabaritos,
            )

    if not questoes:
        st.session_state.messages.append({"role": "assistant", "content": erro_msg, "kind": "chat"})
    else:
        st.session_state.messages.append({
            "role": "assistant",
            "content": sg.para_markdown(dados_simulado),
            "kind": "simulado_json",
            "dados": dados_simulado,
        })
    save_session(st.session_state.session_id, st.session_state.title, st.session_state.messages)
    st.rerun()


# ---------------------------------------------------------------
# Input do usuário
# ---------------------------------------------------------------
user_input = st.chat_input("Pergunte algo sobre seus documentos...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
        render_acoes_mensagem(user_input, key=f"live_user_{len(st.session_state.messages)}")

    # Se for a primeira pergunta da conversa, usa ela pra sugerir um título automático
    if len(st.session_state.messages) == 1 and st.session_state.title == "Nova conversa":
        st.session_state.title = user_input[:50] + ("..." if len(user_input) > 50 else "")

    full_query = build_query_with_context(user_input)

    with st.chat_message("assistant"):
        with st.spinner(f"Consultando (método: {method})..."):
            answer = run_graphrag_query(full_query, method, community_level)
        st.markdown(answer)
        render_acoes_mensagem(answer, key=f"live_assistant_{len(st.session_state.messages)}")

    st.session_state.messages.append({"role": "assistant", "content": answer})

    # Salva em disco a cada troca, para não perder nada se o app for fechado
    save_session(st.session_state.session_id, st.session_state.title, st.session_state.messages)