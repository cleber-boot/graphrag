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

------------------------------------------------------------------
Notas desta revisão (diagnóstico de "nenhuma questão gerada" / JSON inválido)
------------------------------------------------------------------
Havia duas camadas de problema:

  1) Erros na chamada à API eram capturados e relançados com mensagem
     específica (em vez de deixar a exceção genérica do SDK vazar ou, pior,
     ser ignorada). `response_format={"type": "json_object"}` tem fallback
     automático caso não seja aceito pelo modelo/proxy. Isso já estava OK.

  2) O problema real por trás de "O modelo não conseguiu devolver um JSON
     válido mesmo após a correção" em tópicos densos (ex.: lotes de 3
     questões com base_conhecimento + micro_questoes) era TRUNCAMENTO por
     limite de tokens, não JSON malformado de verdade: a resposta batia no
     `max_tokens`, era cortada no meio do objeto, o parse falhava (como
     esperado) e a "correção" pedia pro modelo reescrever o MESMO conteúdo
     no MESMO orçamento de tokens — ou seja, cortava de novo do mesmo jeito.

  Correções aplicadas:
    - `_chamar_llm` agora devolve também o `finish_reason` da resposta.
    - `_gerar_lote` detecta `finish_reason == "length"` (truncamento) e,
      nesse caso, DOBRA o orçamento de tokens e tenta de novo (até um teto),
      em vez de tratar como "JSON malformado" e pedir uma reescrita inútil
      no mesmo orçamento.
    - `MAX_TOKENS_GERACAO` foi aumentado (4000 -> 8000), já que um lote de
      3 questões com base de conhecimento + micro questões facilmente passa
      de 4000 tokens.
    - O tamanho do lote agora é calculado dinamicamente por
      `_tamanho_lote_efetivo`: quando base_conhecimento e/ou micro_questoes
      estão ativados, o lote diminui (menos questões por chamada = menos
      risco de estourar qualquer teto de tokens, e resposta mais
      previsível). Ver essa função para as regras exatas.
    - Quando o JSON volta sem a chave "questoes", a exceção levantada informa
      quais chaves vieram de fato, para facilitar o diagnóstico.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import streamlit as st

# Tamanho do lote "base" (sem base de conhecimento nem micro questões).
# Quando esses extras estão ligados, o lote é reduzido automaticamente
# (ver _tamanho_lote_efetivo), porque cada extra multiplica bastante o
# tamanho da resposta esperada por questão.
TAMANHO_LOTE = 3

# Quantidade máxima de entidades do conhecimento que entram no contexto.
MAX_ENTIDADES_CONTEXTO = 60

# Tokens máximos por resposta do LLM ao gerar questões. Aumentado: um lote
# de 3 questões com base de conhecimento + micro questões pode passar de
# 4000 tokens facilmente (esse era o bug real por trás de "JSON inválido"
# em tópicos densos — o modelo era cortado no meio pelo max_tokens, não
# estava de fato escrevendo JSON malformado). Ver _gerar_lote: se ainda
# assim a resposta for cortada, o orçamento é dobrado automaticamente.
MAX_TOKENS_GERACAO = 8000


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
# 2) Geração das questões (JSON estruturado com Lógica Socrática)
# ---------------------------------------------------------------
PROMPT_JSON = """Você é um elaborador de provas experiente, especializado em reproduzir fielmente o estilo da \
banca {banca} em concursos públicos de Tecnologia da Informação, utilizando rigorosamente o Método Socrático.

MATERIAL DE REFERÊNCIA (extraído da base de conhecimento do candidato):
\"\"\"
{contexto}
\"\"\"

Crie {quantidade} questão(ões) {sobre_tema}no estilo da banca {banca}, usando EXCLUSIVAMENTE conceitos, \
tecnologias, normas e relações presentes no material acima. Não invente nada que não esteja no material.

DIRETRIZES DO MÉTODO SOCRÁTICO PARA AS QUESTÕES:
1. ENUNCIADO PRINCIPAL: Deve descrever um mini-cenário prático de problema de TI (uma falha de arquitetura, um bug de concorrência, uma vulnerabilidade ou um gargalo de infraestrutura) retirado do material. O cenário deve instigar o aluno a deduzir a utilidade ou a consequência técnica da tecnologia correta. NUNCA faça perguntas diretas ou conceituais (ex: proibido "O que é X?" ou "Segundo o texto...").
2. FORMATO DAS ALTERNATIVAS DA QUESTÃO PRINCIPAL: {formato}. Apenas uma alternativa correta deve resolver de fato o problema técnico do cenário apresentado. As outras 4 devem ser distratores plausíveis de TI que geram falhas lógicas ou não resolvem o gargalo do enunciado.

Para CADA questão principal, produza também o JSON contendo os blocos de apoio conforme as instruções abaixo:
{blocos_apoio}
{instrucao_certo_errado}

RESPONDA APENAS COM UM OBJETO JSON VÁLIDO, sem texto antes ou depois, sem cercas de código, exatamente \
neste formato:

{{
  "questoes": [
    {{
      "enunciado": "texto do enunciado prático e baseado em problema da questão principal",
      "alternativas": [
        {{"letra": "A", "texto": "..."}},
        {{"letra": "B", "texto": "..."}}
      ],
      "gabarito": "letra correta (ou CERTO/ERRADO)",
      "comentario": "comentário socrático explicando por que a correta sana o problema prático e o erro lógico de cada distrator, destacando a malícia da banca {banca}",
      "base_conhecimento": [
        {{"titulo": "título curto do conceito", "conteudo": "explicação didática"}}
      ],
      "micro_questoes": [
        {{
          "conceito_alvo": "conceito cobrado nesta micro questão",
          "enunciado": "texto do cenário/provocação da micro questão",
          "alternativas": [
            {{"letra": "A", "texto": "..."}},
            {{"letra": "B", "texto": "..."}},
            {{"letra": "C", "texto": "..."}}
          ],
          "gabarito": "letra correta",
          "explicacao": "explicação da linha de raciocínio dedutivo"
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
2. "micro_questoes": exatamente {n_micro} questões MENORES de múltipla escolha (A, B, C) que aplicam o \
MÉTODO SOCRÁTICO para decompor o raciocínio da questão principal. Cada microquestão NÃO deve perguntar definições \
diretas. Em vez disso, deve colocar o aluno diante de uma micro-provocação ou mini-gargalo técnico isolado da base \
de conhecimento, in ordem crescente de complexidade. Elas devem guiar o raciocínio dedutivo passo a passo do aluno, \
de modo que, ao resolver as alternativas corretas das microquestões, ele descubra por si mesmo a lógica necessária \
para matar a questão principal."""

INSTRUCAO_CERTO_ERRADO = (
    'Como o formato é CERTO ou ERRADO, a questão principal deve ter "alternativas": [] (lista vazia) e '
    '"gabarito" igual a "CERTO" ou "ERRADO". As micro questões continuam com alternativas A, B e C.'
)

# Reforço textual usado quando a chamada precisa ser refeita sem o parâmetro
# response_format (porque o modelo/proxy não o suporta). Deixa explícito
# que a resposta deve ser só o JSON, já que perdemos a garantia estrutural
# que o response_format oferecia.
REFORCO_APENAS_JSON = (
    "\n\nIMPORTANTÍSSIMO: responda SOMENTE com o objeto JSON pedido acima. Não inclua nenhum texto "
    "explicativo antes ou depois, nem cercas de código markdown (```)."
)


class ErroGeracaoQuestoes(RuntimeError):
    """Erro específico para falhas ao gerar questões, com mensagem já
    pronta para ser exibida ao usuário no Streamlit."""


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


def _tamanho_lote_efetivo(incluir_base: bool, incluir_micro: bool, n_micro: int) -> int:
    """Reduz o lote quando a resposta por questão fica maior (base de
    conhecimento e/ou micro questões), para não depender só de aumentar
    max_tokens — menos itens por chamada também reduz o risco de
    truncamento, e mantém os lotes mais previsíveis.

    Regras (empíricas, ajuste se notar truncamento mesmo assim):
      - nem base nem micro: lote cheio (TAMANHO_LOTE, hoje 3)
      - só um dos dois: lote médio (2)
      - os dois juntos: lote pequeno (2), e menor ainda (1) se n_micro for alto
    """
    if not incluir_base and not incluir_micro:
        return TAMANHO_LOTE
    if incluir_base and incluir_micro:
        return 1 if n_micro >= 4 else 2
    return 2  # só base OU só micro


def _chamar_llm(
    client,
    modelo: str,
    prompt: str,
    *,
    usar_response_format: bool,
    max_tokens: int = MAX_TOKENS_GERACAO,
) -> tuple[str, str | None]:
    """Chama a API do LLM, isolando os kwargs para facilitar o fallback
    sem response_format quando o modelo/proxy não o suporta.

    Devolve (conteudo, finish_reason). O finish_reason é usado por quem
    chama para distinguir "resposta cortada por limite de tokens"
    (finish_reason == "length") de "resposta malformada por outro motivo",
    já que os dois casos pedem correções bem diferentes.
    """
    kwargs = dict(
        model=modelo,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=max_tokens,
    )
    if usar_response_format:
        kwargs["response_format"] = {"type": "json_object"}

    resposta = client.chat.completions.create(**kwargs)
    escolha = resposta.choices[0]
    return escolha.message.content, getattr(escolha, "finish_reason", None)


def _gerar_lote(client, modelo: str, prompt: str, *, tentativa_max_tokens: int = MAX_TOKENS_GERACAO) -> dict:
    """Faz a chamada ao LLM e devolve o dict já parseado, tentando:
    1) com response_format;
    2) se falhar a CHAMADA em si (ex.: parâmetro não suportado), sem response_format;
    3) se a resposta veio CORTADA por limite de tokens (finish_reason ==
       "length"), dobra o orçamento de max_tokens e tenta de novo — pedir
       para "reescrever como JSON válido" no mesmo orçamento não resolveria,
       porque o conteúdo pedido é maior do que o espaço disponível;
    4) se o JSON vier malformado por outro motivo (não truncamento), pede
       pro modelo reescrever só o JSON.

    Levanta ErroGeracaoQuestoes com uma mensagem específica se nada funcionar.
    """
    conteudo = None
    finish_reason = None
    erro_chamada = None

    try:
        conteudo, finish_reason = _chamar_llm(
            client, modelo, prompt, usar_response_format=True, max_tokens=tentativa_max_tokens
        )
    except Exception as exc:  # noqa: BLE001 - queremos capturar qualquer falha do SDK/proxy
        erro_chamada = exc
        try:
            conteudo, finish_reason = _chamar_llm(
                client, modelo, prompt + REFORCO_APENAS_JSON,
                usar_response_format=False, max_tokens=tentativa_max_tokens,
            )
        except Exception as exc2:  # noqa: BLE001
            raise ErroGeracaoQuestoes(
                "Falha ao chamar o modelo de linguagem. Primeira tentativa "
                f"(com response_format=json_object): {erro_chamada}. "
                f"Segunda tentativa (sem response_format): {exc2}."
            ) from exc2

    # Cortado por limite de tokens: NÃO adianta pedir pro modelo "reescrever
    # como JSON válido" no mesmo orçamento — ele vai cortar de novo no mesmo
    # lugar. Dobra o orçamento (até um teto) e tenta de novo do zero.
    if finish_reason == "length":
        if tentativa_max_tokens < MAX_TOKENS_GERACAO * 2:
            return _gerar_lote(client, modelo, prompt, tentativa_max_tokens=tentativa_max_tokens * 2)
        raise ErroGeracaoQuestoes(
            "A resposta do modelo foi cortada por limite de tokens mesmo após "
            f"dobrar o orçamento para {tentativa_max_tokens}. Reduza o lote "
            "(TAMANHO_LOTE / _tamanho_lote_efetivo), o número de micro questões "
            "(n_micro), ou desative a base de conhecimento/micro questões para "
            f"este conhecimento. Início da resposta cortada: {(conteudo or '')[:300]!r}"
        )

    try:
        return _extrair_json(conteudo)
    except ValueError:
        # Uma segunda tentativa, pedindo só o JSON de volta (aqui o motivo NÃO
        # foi truncamento, então faz sentido pedir uma reescrita).
        try:
            correcao_conteudo, correcao_finish = _chamar_llm(
                client, modelo,
                prompt + f"\n\n[RESPOSTA ANTERIOR PARA CORRIGIR]\n{conteudo}\n\n"
                "Reescreva a resposta anterior como JSON válido, sem nenhum texto fora do objeto JSON.",
                usar_response_format=False, max_tokens=tentativa_max_tokens,
            )
        except Exception as exc:  # noqa: BLE001
            raise ErroGeracaoQuestoes(
                f"O modelo devolveu um JSON inválido e a tentativa de correção também falhou: {exc}. "
                f"Início da resposta original: {(conteudo or '')[:300]!r}"
            ) from exc

        if correcao_finish == "length":
            raise ErroGeracaoQuestoes(
                "A correção também foi cortada por limite de tokens — o conteúdo "
                "pedido é maior do que o orçamento atual. Reduza TAMANHO_LOTE "
                "(ou n_micro) e tente novamente."
            )

        try:
            return _extrair_json(correcao_conteudo)
        except ValueError as exc:
            raise ErroGeracaoQuestoes(
                "O modelo não conseguiu devolver um JSON válido mesmo após a correção. "
                f"Início da resposta: {(correcao_conteudo or '')[:300]!r}"
            ) from exc


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
    """Gera `quantidade` questões em lotes pequenos e devolve a lista de dicts.

    O tamanho do lote é calculado dinamicamente (ver _tamanho_lote_efetivo):
    quanto mais "pesada" fica cada questão (base de conhecimento + micro
    questões), menor o lote, para reduzir o risco de a resposta do modelo
    ser cortada por limite de tokens. Se mesmo assim uma resposta for
    cortada, `_gerar_lote` dobra automaticamente o orçamento de tokens
    antes de desistir.

    Levanta ErroGeracaoQuestoes (com mensagem específica) se algum lote
    falhar de forma irrecuperável. Antes essa função apenas retornava uma
    lista (possivelmente vazia) sem explicar o motivo — o que fazia o
    chat_app.py mostrar apenas "Nenhuma questão foi gerada", sem pista
    nenhuma da causa real.
    """
    questoes: list[dict] = []
    restante = max(1, int(quantidade))
    tamanho_lote = _tamanho_lote_efetivo(incluir_base, incluir_micro, int(n_micro))

    apoio = ""
    if incluir_base:
        apoio += INSTRUCAO_BASE
    if incluir_micro:
        apoio += "\n" + INSTRUCAO_MICRO.format(n_micro=max(1, int(n_micro)))
    if not apoio:
        apoio = '\nDeixe "base_conhecimento" e "micro_questoes" como listas vazias.'

    if not contexto or not contexto.strip():
        raise ErroGeracaoQuestoes(
            "O contexto enviado ao modelo está vazio. Verifique se a comunidade "
            "selecionada tem 'full_content'/'summary' preenchidos no "
            "community_reports.parquet, e se communities.parquet/entities.parquet "
            "existem e estão atualizados."
        )

    while restante > 0:
        lote = min(tamanho_lote, restante)
        prompt = PROMPT_JSON.format(
            banca=banca.strip() or "FGV",
            contexto=contexto,
            quantidade=lote,
            sobre_tema=f"sobre '{tema.strip()}' " if tema.strip() else "",
            formato=formato_descricao,
            blocos_apoio=apoio,
            instrucao_certo_errado=INSTRUCAO_CERTO_ERRADO if certo_errado else "",
        )

        dados = _gerar_lote(client, modelo, prompt)

        novas = dados.get("questoes") or []
        if not isinstance(novas, list) or not novas:
            raise ErroGeracaoQuestoes(
                "O modelo devolveu um JSON válido, mas sem a chave 'questoes' "
                f"preenchida como lista. Chaves recebidas: {list(dados.keys())!r}."
            )
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
    """Versão em texto do simulado — usada como content da mensagem salva,
    para que buscas e exportações continuem funcionando."""
    linhas = [dados.get("cabecalho", "")]
    for i, q in enumerate(dados.get("questoes", []), start=1):
        linhas.append(f"## Questão {i}")
        if q.get("_conhecimento"):
            linhas.append(f"Conhecimento: {q['_conhecimento']}")
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