# 🧠 GraphRAG — Cérebro de Estudos de TI

Assistente de estudos pessoal baseado em **GraphRAG** (Microsoft) para transformar material de TI (PDFs de aulas, resumos, apostilas) em uma **base de conhecimento em grafo**, consultável via chat — com geração de **simulados no estilo de bancas de concurso**.

O projeto foi pensado para quem estuda para concursos/certificações de TI: você joga PDFs de aula na pasta de entrada, o pipeline extrai conceitos, tabelas, figuras e relações técnicas, o GraphRAG indexa tudo em um grafo de conhecimento, e você conversa com esse conhecimento (ou pede para gerar questões) por uma interface de chat em Streamlit.

## ✨ Funcionalidades

- **Extração estruturada de PDFs** (`pipeline_completo_ti.py`): usa um LLM multimodal (via OpenRouter) para ler PDFs de aula e extrair, em JSON estruturado (Pydantic):
  - Metadados do documento (título, tecnologias citadas, idioma)
  - Conceitos-chave com definição, tipo, vantagens/desvantagens e casos de uso
  - Tabelas comparativas
  - Figuras/diagramas com descrição técnica
  - Relações entre entidades (`depende_de`, `implementa`, `mitiga_risco`, `substitui`, `comunica_com`, etc.)
- **Indexação em grafo** com o [GraphRAG](https://microsoft.github.io/graphrag/) da Microsoft, permitindo buscas `local`, `global` e `drift`.
- **Chat com memória persistente** (`chat_app.py`): interface Streamlit que roda `graphrag query` por baixo dos panos e salva cada conversa em disco (`chat_sessions/*.json`), permitindo retomar de onde parou.
- **Gerador de simulados**: cria questões no estilo de uma banca específica (ex: FGV, Cesgranrio, Cebraspe), usando exclusivamente o conteúdo já indexado como fonte de verdade.

## 📁 Estrutura do repositório

```
graphrag/
└── cerebro_estudos_ti/
    ├── pipeline_completo_ti.py   # Extrai conteúdo estruturado dos PDFs de entrada
    ├── chat_app.py               # Interface de chat (Streamlit) sobre o índice GraphRAG
    ├── settings.yaml             # Configuração do GraphRAG (modelos, chunking, etc.)
    ├── input/                    # Textos já extraídos, prontos para indexação
    ├── arquivos_processados/     # PDFs originais já processados pelo pipeline
    ├── prompts/                  # Prompts usados pelo GraphRAG em cada etapa
    ├── output/                   # Índice final (parquet + LanceDB) gerado pelo GraphRAG
    ├── update_output/            # Snapshots de atualizações incrementais do índice
    └── logs/                     # Logs de indexação e consultas
```

## 🚀 Como usar

### 1. Pré-requisitos

- Python 3.10+
- Uma chave de API da [OpenRouter](https://openrouter.ai/) (usada tanto para o pipeline de extração quanto para os modelos do GraphRAG)

### 2. Instalação

```bash
git clone https://github.com/cleber-boot/graphrag.git
cd graphrag/cerebro_estudos_ti

pip install graphrag streamlit openai pydantic python-dotenv json-repair
```

Crie um arquivo `.env` dentro de `cerebro_estudos_ti/` com sua chave:

```
GRAPHRAG_API_KEY=sua_chave_openrouter_aqui
```

### 3. Processar novos materiais de estudo

Coloque os PDFs das suas aulas/resumos em uma pasta de entrada e rode:

```bash
python pipeline_completo_ti.py
```

O script extrai o conteúdo estruturado de cada PDF e gera os arquivos de texto usados pelo GraphRAG em `input/`, movendo os PDFs já processados para `arquivos_processados/`.

### 4. (Primeira vez) Inicializar o projeto GraphRAG

Se ainda não existir `settings.yaml`/`prompts/` (ou para começar um novo projeto do zero), inicialize a estrutura:

```bash
python -m graphrag init --root .
```

### 5. (Opcional, mas recomendado) Ajustar os prompts ao domínio

O GraphRAG usa prompts genéricos por padrão. Para melhorar a qualidade da extração de entidades/relações no domínio de estudos de TI, gere prompts calibrados automaticamente com base no seu próprio conteúdo:

```bash
python -m graphrag prompt-tune --root . --domain "Information Technology and Computer Science Exams" --language "Portuguese"
```

Isso sobrescreve os arquivos em `prompts/` com versões ajustadas ao domínio e ao idioma do material (português).

### 6. Indexar a base de conhecimento

```bash
python -m graphrag index --root .
```

Isso lê os arquivos em `input/`, aplica os prompts em `prompts/` e constrói o grafo de conhecimento em `output/`.

### 7. Atualizar o índice incrementalmente

Depois de rodar `pipeline_completo_ti.py` novamente para processar novos PDFs, não é necessário reindexar tudo do zero — use o comando de atualização incremental, que reaproveita o índice existente e registra o resultado em `update_output/`:

```bash
python -m graphrag update --root .
```

### 8. Conversar com a base de conhecimento

Via linha de comando:

```bash
python -m graphrag query --root . --method local --query "O que é idempotência?"
```

Ou pela interface de chat:

```bash
streamlit run chat_app.py
```

A interface permite escolher o método de busca (`local`, `global` ou `drift`), manter histórico de conversas e gerar simulados de questões no estilo de bancas específicas, a partir do conteúdo indexado.

## ⚙️ Configuração

O arquivo `settings.yaml` define os modelos usados (por padrão, `google/gemini-2.5-flash-lite` para completions e `google/gemini-embedding-2` para embeddings, ambos via OpenRouter) e os parâmetros de chunking dos documentos. Ajuste conforme sua necessidade e limites de API.

## 📝 Licença

Projeto pessoal de estudos. Adapte livremente para seu próprio fluxo de aprendizado.
