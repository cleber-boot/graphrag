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
from datetime import datetime
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------
ROOT_DIR = "./cerebro_estudos_ti"
SESSIONS_DIR = Path(ROOT_DIR) / "chat_sessions"
SESSIONS_DIR.mkdir(exist_ok=True)

MAX_HISTORY_TURNS = 4
TIMEOUT_SECONDS = 300

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

Com base no conteúdo do material indexado (sua base de conhecimento), crie um simulado com {quantidade} questões \
{sobre_tema}seguindo rigorosamente as características da banca {banca}:
- Reproduza o nível de dificuldade, a forma de redigir o enunciado e o estilo de pegadinha típicos dessa banca.
- Use apenas conceitos, tecnologias e relações que estejam de fato presentes na base de conhecimento.

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

st.set_page_config(page_title="Chat GraphRAG", page_icon="🧠", layout="centered")


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
    tema_simulado = st.text_input("Tema (opcional)", placeholder="Ex: redes de computadores, segurança da informação...")
    quantidade_questoes = st.number_input("Quantidade de questões", min_value=1, max_value=20, value=5, step=1)
    formato_questao = st.selectbox("Formato da questão", options=list(FORMATOS_QUESTAO.keys()))

    gerar_simulado_clicado = st.button("🎯 Gerar simulado agora", use_container_width=True, type="primary")

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


def build_simulado_prompt(banca: str, tema: str, quantidade: int, formato_label: str) -> str:
    sobre_tema = f"sobre o tema '{tema}' " if tema.strip() else ""
    return PROMPT_SIMULADO.format(
        banca=banca.strip() or "FGV",
        quantidade=quantidade,
        sobre_tema=sobre_tema,
        formato_alternativas=FORMATOS_QUESTAO[formato_label],
    )


def formatar_simulado(texto: str) -> str:
    """Rede de segurança: reformata em Markdown mesmo se o modelo devolver
    tudo (ou parte) em bloco de texto corrido. Roda sempre, e os padrões usam
    lookbehind para não bagunçar trechos que já estão formatados corretamente
    (evita duplicar '##', '**', etc. em questões que já vieram certas)."""

    # Quebra antes de cada "Questão N" (aceita maiúsc./minúsc.), exceto se já tiver "## " na frente
    texto = re.sub(
        r"(?<!#\s)[Qq]uest(?:ã|a)o\s+(\d+)\b\s*",
        r"\n\n---\n\n## Questão \1\n\n",
        texto,
    )
    # Quebra antes de cada alternativa A) a E), exceto se já estiver em negrito ("**A)**")
    texto = re.sub(r"(?<!\*)\s+([A-E])\)\s+", r"\n\n**\1)** ", texto)
    # Destaca o gabarito em negrito, em linha própria (aceita "Gabarito:" ou "GABARITO:")
    texto = re.sub(r"(?<!\*)\s*[Gg][Aa][Bb][Aa][Rr][Ii][Tt][Oo]:\s*", r"\n\n**Gabarito:** ", texto)
    # Formata o comentário como citação em bloco (aceita "Comentário:" ou "comentário:")
    texto = re.sub(r"(?<!\*)\s*[Cc]oment[áa]rio:\s*", r"\n\n> **Comentário:** ", texto)

    return texto.strip()


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
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------------------------------------------------------
# Geração de simulado (via botão na barra lateral)
# ---------------------------------------------------------------
if gerar_simulado_clicado:
    tema_label = tema_simulado.strip() if tema_simulado.strip() else "conteúdo geral da base"
    pedido_visivel = f"🎯 Gerar simulado — banca {banca or 'FGV'} · {quantidade_questoes} questões · {tema_label}"

    st.session_state.messages.append({"role": "user", "content": pedido_visivel})
    with st.chat_message("user"):
        st.markdown(pedido_visivel)

    if len(st.session_state.messages) == 1 and st.session_state.title == "Nova conversa":
        st.session_state.title = f"Simulado {banca or 'FGV'} — {tema_label}"[:50]

    prompt_simulado = build_simulado_prompt(banca, tema_simulado, quantidade_questoes, formato_questao)

    # Simulados se beneficiam de contexto amplo do material -> usa 'global' por padrão,
    # independente do método selecionado para o chat normal, a menos que o usuário já esteja em 'drift'.
    metodo_para_simulado = "drift" if method == "drift" else "global"

    with st.chat_message("assistant"):
        with st.spinner(f"Gerando simulado (método: {metodo_para_simulado})... isso pode levar um tempo."):
            resposta_bruta = run_graphrag_query(prompt_simulado, metodo_para_simulado, community_level)
            resposta_simulado = formatar_simulado(resposta_bruta)
        st.markdown(resposta_simulado)

    st.session_state.messages.append({"role": "assistant", "content": resposta_simulado})
    save_session(st.session_state.session_id, st.session_state.title, st.session_state.messages)


# ---------------------------------------------------------------
# Input do usuário
# ---------------------------------------------------------------
user_input = st.chat_input("Pergunte algo sobre seus documentos...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Se for a primeira pergunta da conversa, usa ela pra sugerir um título automático
    if len(st.session_state.messages) == 1 and st.session_state.title == "Nova conversa":
        st.session_state.title = user_input[:50] + ("..." if len(user_input) > 50 else "")

    full_query = build_query_with_context(user_input)

    with st.chat_message("assistant"):
        with st.spinner(f"Consultando (método: {method})..."):
            answer = run_graphrag_query(full_query, method, community_level)
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

    # Salva em disco a cada troca, para não perder nada se o app for fechado
    save_session(st.session_state.session_id, st.session_state.title, st.session_state.messages)