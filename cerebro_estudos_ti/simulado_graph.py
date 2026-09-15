"""
Simulados baseados nos CONHECIMENTOS do grafo (cerebro_estudos_ti)
------------------------------------------------------------------
Este módulo cuida de três coisas que o chat_app.py apenas consome:

1. Ler as comunidades do GraphRAG (output/community_reports.parquet) e
   expô-las como "conhecimentos" selecionáveis, para que o usuário escolha
   QUANTAS questões quer de CADA conhecimento.

2. Gerar as questões em JSON estruturado (em vez de Markdown solto), o que
   permite que cada questão principal carregue:
      - base_conhecimento : blocos de teoria necessários para resolvê-la
      - micro_questoes    : questões menores que decompõem o raciocínio
      - gabarito/comentário

3. Renderizar tudo no Streamlit com a base de conhecimento, as micro questões
   e o gabarito em expansores COLAPSADOS por padrão.

Observação sobre expansores: o Streamlit não permite expander dentro de
expander. Por isso as respostas das micro questões usam checkbox dentro do
expander do grupo, e não outro expander.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import streamlit as st

# Quantas questões pedir por chamada de LLM. Lotes pequenos deixam o JSON
# mais confiável e evitam resposta truncada quando há micro questões.
TAMANHO_LOTE = 3

# Quantidade máxima de entidades do conhecimento que entram no contexto.
MAX_ENTIDADES_CONTEXTO = 60


# ---------------------------------------------------------------
# 1) Leitura dos conhecimentos do grafo
# ---------------------------------------------------------------
def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


@st.cache_data(show_spinner=False)
def carregar_conhecimentos(root_dir: str, _assinatura: float = 0.0) -> pd.DataFrame:
    """Lê os relatórios de comunidade e devolve um DataFrame de 'conhecimentos'.

    Cada comunidade do GraphRAG é um agrupamento temático de entidades — é
    exatamente o que o usuário enxerga como 'um conhecimento do grafo'.
    """
    caminho = Path(root_dir) / "output" / "community_reports.parquet"
    if not caminho.exists():
        return pd.DataFrame()

    df = pd.read_parquet(caminho)
    colunas = [c for c in
               ["community", "level", "title", "summary", "full_content", "rank", "size", "findings"]
               if c in df.columns]
    df = df[colunas].copy()

    if "size" not in df.columns:
        df["size"] = 0
    if "rank" not in df.columns:
        df["rank"] = 0.0

    df["community"] = df["community"].astype(int)
    df["level"] = df["level"].astype(int)
    df["size"] = df["size"].fillna(0).astype(int)
    df = df.sort_values(["level", "size"], ascending=[True, False]).reset_index(drop=True)
    df["rotulo"] = df.apply(
        lambda r: f"[N{r['level']}] {str(r['title']).strip()} · {r['size']} entidades (#{r['community']})",
        axis=1,
    )
    return df


@st.cache_data(show_spinner=False)
def carregar_entidades_por_comunidade(root_dir: str, _assinatura: float = 0.0) -> dict[int, list[str]]:
    """Mapa {community -> lista de 'NOME: descrição'}, usado para enriquecer o
    contexto de cada conhecimento com detalhes técnicos concretos."""
    dir_out = Path(root_dir) / "output"
    p_com = dir_out / "communities.parquet"
    p_ent = dir_out / "entities.parquet"
    if not (p_com.exists() and p_ent.exists()):
        return {}

    comunidades = pd.read_parquet(p_com)
    entidades = pd.read_parquet(p_ent)

    descricao_por_id: dict[str, str] = {}
    for _, e in entidades.iterrows():
        titulo = str(e.get("title", "")).strip()
        desc = str(e.get("description", "") or "").strip()
        if titulo:
            descricao_por_id[str(e.get("id"))] = f"{titulo}: {desc}" if desc else titulo

    mapa: dict[int, list[str]] = {}
    for _, c in comunidades.iterrows():
        ids = c.get("entity_ids")
        if ids is None:
            continue
        linhas = [descricao_por_id[str(i)] for i in list(ids) if str(i) in descricao_por_id]
        mapa[int(c["community"])] = linhas[:MAX_ENTIDADES_CONTEXTO]
    return mapa


def assinatura_indice(root_dir: str) -> float:
    """Usada como chave de cache: muda quando o índice é regerado."""
    dir_out = Path(root_dir) / "output"
    return max(
        _mtime(dir_out / "community_reports.parquet"),
        _mtime(dir_out / "communities.parquet"),
        _mtime(dir_out / "entities.parquet"),
    )


def filtrar_conhecimentos(df: pd.DataFrame, niveis: list[int], busca: str) -> pd.DataFrame:
    if df.empty:
        return df
    filtrado = df[df["level"].isin(niveis)] if niveis else df
    termo = (busca or "").strip().lower()
    if termo:
        alvo = (
            filtrado["title"].astype(str).str.lower()
            + " "
            + filtrado["summary"].astype(str).str.lower()
        )
        filtrado = filtrado[alvo.str.contains(re.escape(termo), na=False)]
    return filtrado


def montar_contexto(linha: pd.Series, entidades: list[str]) -> str:
    """Contexto enviado ao LLM para um conhecimento específico."""
    partes = [f"CONHECIMENTO: {linha.get('title', '')}"]

    conteudo = str(linha.get("full_content") or linha.get("summary") or "").strip()
    if conteudo:
        partes.append(conteudo)

    if entidades:
        partes.append(
            "ENTIDADES E DEFINIÇÕES TÉCNICAS DESTE CONHECIMENTO:\n"
            + "\n".join(f"- {linha_ent}" for linha_ent in entidades)
        )
    return "\n\n".join(partes)


# ---------------------------------------------------------------
# 2) Geração das questões (JSON estruturado)
# ---------------------------------------------------------------
PROMPT_JSON = """Você é um elaborador de provas experiente, especializado em reproduzir fielmente o estilo da \
banca {banca} em concursos públicos de Tecnologia da Informação.

MATERIAL DE REFERÊNCIA (extraído da base de conhecimento do candidato):
\"\"\"
{contexto}
\"\"\"

Crie {quantidade} questão(ões) {sobre_tema}no estilo da banca {banca}, usando EXCLUSIVAMENTE conceitos, \
tecnologias, normas e relações presentes no material acima. Não invente nada que não esteja no material.

FORMATO DAS ALTERNATIVAS DA QUESTÃO PRINCIPAL: {formato}

Para CADA questão principal, produza também:
{blocos_apoio}
{instrucao_certo_errado}

RESPONDA APENAS COM UM OBJETO JSON VÁLIDO, sem texto antes ou depois, sem cercas de código, exatamente \
neste formato:

{{
  "questoes": [
    {{
      "enunciado": "texto do enunciado da questão principal",
      "alternativas": [
        {{"letra": "A", "texto": "..."}},
        {{"letra": "B", "texto": "..."}}
      ],
      "gabarito": "letra correta (ou CERTO/ERRADO)",
      "comentario": "por que a correta está certa e por que cada uma das outras está errada, destacando a pegadinha típica da banca {banca}",
      "base_conhecimento": [
        {{"titulo": "título curto do conceito", "conteudo": "explicação didática"}}
      ],
      "micro_questoes": [
        {{
          "conceito_alvo": "conceito cobrado nesta micro questão",
          "enunciado": "texto da micro questão",
          "alternativas": [
            {{"letra": "A", "texto": "..."}},
            {{"letra": "B", "texto": "..."}},
            {{"letra": "C", "texto": "..."}}
          ],
          "gabarito": "letra correta",
          "explicacao": "explicação curta"
        }}
      ]
    }}
  ]
}}

Todo o conteúdo deve estar em português do Brasil. Use aspas duplas em todas as chaves e valores e escape \
corretamente aspas internas."""

INSTRUCAO_BASE = """
1. "base_conhecimento": de 2 a 4 blocos curtos de teoria, retirados do material, contendo TUDO o que o \
candidato precisa saber para conseguir responder a questão principal por conta própria. Cada bloco tem um \
título curto e uma explicação didática de 2 a 5 frases. Escreva como material de estudo, não como resposta: \
NUNCA revele qual é a alternativa correta dentro da base de conhecimento. Quando a questão principal for \
complexa, divida o conteúdo em blocos menores, cada um tratando de uma parte do raciocínio."""

INSTRUCAO_MICRO = """
2. "micro_questoes": exatamente {n_micro} questões MENORES e mais simples que decompõem o raciocínio da \
questão principal. Cada micro questão deve cobrar um único conceito isolado da base de conhecimento, em \
ordem crescente de dificuldade, de modo que quem acertar todas tenha construído o raciocínio completo \
necessário para resolver a questão principal. Use 3 alternativas (A, B, C) nas micro questões, com \
explicação curta da resposta."""

INSTRUCAO_CERTO_ERRADO = (
    'Como o formato é CERTO ou ERRADO, a questão principal deve ter "alternativas": [] (lista vazia) e '
    '"gabarito" igual a "CERTO" ou "ERRADO". As micro questões continuam com alternativas A, B e C.'
)


def _extrair_json(bruto: str) -> dict:
    """Tolera cercas de código e texto solto ao redor do JSON."""
    texto = (bruto or "").strip()
    texto = re.sub(r"^```(?:json)?\s*", "", texto)
    texto = re.sub(r"\s*```$", "", texto)

    try:
        return json.loads(texto)
    except json.JSONDecodeError:
        pass

    inicio = texto.find("{")
    fim = texto.rfind("}")
    if inicio != -1 and fim > inicio:
        try:
            return json.loads(texto[inicio:fim + 1])
        except json.JSONDecodeError:
            pass
    raise ValueError("O modelo não devolveu um JSON válido.")


def gerar_questoes(
    client,
    modelo: str,
    contexto: str,
    banca: str,
    tema: str,
    quantidade: int,
    formato_descricao: str,
    certo_errado: bool,
    n_micro: int,
    incluir_base: bool = True,
    incluir_micro: bool = True,
) -> list[dict]:
    """Gera `quantidade` questões em lotes pequenos e devolve a lista de dicts."""
    questoes: list[dict] = []
    restante = max(1, int(quantidade))

    apoio = ""
    if incluir_base:
        apoio += INSTRUCAO_BASE
    if incluir_micro:
        apoio += "\n" + INSTRUCAO_MICRO.format(n_micro=max(1, int(n_micro)))
    if not apoio:
        apoio = '\nDeixe "base_conhecimento" e "micro_questoes" como listas vazias.'

    while restante > 0:
        lote = min(TAMANHO_LOTE, restante)
        prompt = PROMPT_JSON.format(
            banca=banca.strip() or "FGV",
            contexto=contexto,
            quantidade=lote,
            sobre_tema=f"sobre '{tema.strip()}' " if tema.strip() else "",
            formato=formato_descricao,
            blocos_apoio=apoio,
            instrucao_certo_errado=INSTRUCAO_CERTO_ERRADO if certo_errado else "",
        )

        resposta = client.chat.completions.create(
            model=modelo,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        conteudo = resposta.choices[0].message.content

        try:
            dados = _extrair_json(conteudo)
        except ValueError:
            # Uma segunda tentativa, pedindo só o JSON de volta.
            correcao = client.chat.completions.create(
                model=modelo,
                messages=[
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": conteudo},
                    {"role": "user", "content": "Reescreva a resposta anterior como JSON válido, sem nenhum texto fora do objeto JSON."},
                ],
                temperature=0,
            )
            dados = _extrair_json(correcao.choices[0].message.content)

        novas = dados.get("questoes") or []
        if not isinstance(novas, list) or not novas:
            break
        questoes.extend(novas[:lote])
        restante -= lote

    return questoes


# ---------------------------------------------------------------
# 3) Renderização no Streamlit
# ---------------------------------------------------------------
def _bloco_alternativas(alternativas) -> str:
    linhas = []
    for alt in alternativas or []:
        letra = str(alt.get("letra", "")).strip().rstrip(")")
        linhas.append(f"**{letra})** {str(alt.get('texto', '')).strip()}")
    return "\n\n".join(linhas)


def render_questao(
    questao: dict,
    numero: int,
    prefixo: str,
    mostrar_base: bool,
    mostrar_micro: bool,
    mostrar_gabarito: bool,
) -> None:
    """Renderiza uma questão principal com seus três blocos colapsáveis."""
    st.markdown(f"## Questão {numero}")

    conhecimento = questao.get("_conhecimento")
    if conhecimento:
        st.caption(f"🧠 Conhecimento do grafo: {conhecimento}")

    st.markdown(str(questao.get("enunciado", "")).strip())

    alternativas = questao.get("alternativas") or []
    if alternativas:
        st.markdown(_bloco_alternativas(alternativas))
    else:
        st.markdown("_Julgue a afirmação acima como **CERTO** ou **ERRADO**._")

    # --- Base de conhecimento (colapsada, igual ao gabarito) -------------
    base = questao.get("base_conhecimento") or []
    if base:
        with st.expander(
            f"📚 Base de conhecimento ({len(base)} conceitos)",
            expanded=mostrar_base,
            key=f"{prefixo}_base_{numero}_{mostrar_base}",
        ):
            for bloco in base:
                titulo = str(bloco.get("titulo", "")).strip()
                if titulo:
                    st.markdown(f"**{titulo}**")
                st.markdown(str(bloco.get("conteudo", "")).strip())
                st.markdown("")

    # --- Micro questões (colapsadas) ------------------------------------
    micros = questao.get("micro_questoes") or []
    if micros:
        with st.expander(
            f"🧩 Micro questões ({len(micros)}) — construa o raciocínio passo a passo",
            expanded=mostrar_micro,
            key=f"{prefixo}_micro_{numero}_{mostrar_micro}",
        ):
            for j, micro in enumerate(micros, start=1):
                alvo = str(micro.get("conceito_alvo", "")).strip()
                if alvo:
                    st.caption(f"Conceito: {alvo}")
                st.markdown(f"**{numero}.{j}** {str(micro.get('enunciado', '')).strip()}")
                st.markdown(_bloco_alternativas(micro.get("alternativas")))

                ver = st.checkbox(
                    "👁️ Ver resposta",
                    value=mostrar_gabarito,
                    key=f"{prefixo}_microresp_{numero}_{j}_{mostrar_gabarito}",
                )
                if ver:
                    st.success(
                        f"**Resposta:** {micro.get('gabarito', '—')}\n\n"
                        f"{str(micro.get('explicacao', '')).strip()}"
                    )
                if j < len(micros):
                    st.divider()

    # --- Gabarito da questão principal (colapsado) ----------------------
    with st.expander(
        "✅ Ver gabarito e comentário",
        expanded=mostrar_gabarito,
        key=f"{prefixo}_gab_{numero}_{mostrar_gabarito}",
    ):
        st.markdown(f"**Gabarito:** {questao.get('gabarito', '—')}")
        st.markdown(f"> **Comentário:** {str(questao.get('comentario', '')).strip()}")

    st.markdown("---")


def render_simulado_estruturado(
    dados: dict,
    prefixo: str,
    mostrar_base: bool,
    mostrar_micro: bool,
    mostrar_gabarito: bool,
) -> None:
    cabecalho = dados.get("cabecalho")
    if cabecalho:
        st.markdown(cabecalho)

    for i, questao in enumerate(dados.get("questoes", []), start=1):
        render_questao(questao, i, prefixo, mostrar_base, mostrar_micro, mostrar_gabarito)


def para_markdown(dados: dict) -> str:
    """Versão em texto do simulado — usada como `content` da mensagem salva,
    para que buscas e exportações continuem funcionando."""
    linhas = [dados.get("cabecalho", "")]
    for i, q in enumerate(dados.get("questoes", []), start=1):
        linhas.append(f"## Questão {i}")
        if q.get("_conhecimento"):
            linhas.append(f"*Conhecimento: {q['_conhecimento']}*")
        linhas.append(str(q.get("enunciado", "")))
        linhas.append(_bloco_alternativas(q.get("alternativas")))
        for bloco in q.get("base_conhecimento") or []:
            linhas.append(f"**{bloco.get('titulo', '')}** — {bloco.get('conteudo', '')}")
        for j, m in enumerate(q.get("micro_questoes") or [], start=1):
            linhas.append(f"**{i}.{j}** {m.get('enunciado', '')}")
            linhas.append(_bloco_alternativas(m.get("alternativas")))
            linhas.append(f"Resposta: {m.get('gabarito', '')} — {m.get('explicacao', '')}")
        linhas.append(f"**Gabarito:** {q.get('gabarito', '')}")
        linhas.append(f"> **Comentário:** {q.get('comentario', '')}")
        linhas.append("---")
    return "\n\n".join(p for p in linhas if p)