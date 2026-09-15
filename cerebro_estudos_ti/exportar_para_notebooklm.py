import os
import re
import pandas as pd
from pathlib import Path
import math

# Pasta onde os arquivos densos serão salvos
PASTA_SAIDA = "conteudo_profundo_notebooklm"
MAX_ARQUIVOS = 49

def limpar_nome_arquivo(nome):
    return re.sub(r'[\\/*?:"<>|]', "", str(nome)).strip().replace(" ", "_")[:50]

def localizar_artefatos():
    print("🔍 Procurando arquivos densos do GraphRAG em output/...")
    output_dir = Path("output")
    
    if not output_dir.exists():
        raise FileNotFoundError("A pasta 'output/' não foi encontrada. Rode a indexação primeiro.")
        
    def buscar(padroes):
        for p in padroes:
            encontrados = list(output_dir.rglob(p))
            if encontrados: 
                return sorted(encontrados)[-1]
        return None

    path_entities = buscar(["*create_final_entities.parquet", "*entities.parquet"])
    path_text_units = buscar(["*create_final_text_units.parquet", "*text_units.parquet"])
    
    return path_entities, path_text_units

def exportar_conteudo_profundo():
    if not os.path.exists(PASTA_SAIDA):
        os.makedirs(PASTA_SAIDA)
    else:
        # Limpa a pasta anterior para não acumular sujeira
        for f in Path(PASTA_SAIDA).glob("*.txt"):
            f.unlink()

    try:
        p_ent, p_tx = localizar_artefatos()
        
        if not p_ent or not p_tx:
            print(f"\n❌ Erro: Não encontrei as tabelas necessárias.\nEntities: {p_ent} | Text Units: {p_tx}")
            return

        print(f"📦 Carregando: \n - {p_ent.name}\n - {p_tx.name}")
        df_entities = pd.read_parquet(p_ent)
        df_text_units = pd.read_parquet(p_tx)

        print("🧠 Coletando toda a base de conhecimentos de TI...")

        # Mapeamento estrito e seguro das colunas (garantindo strings)
        col_ent_id = "id" if "id" in df_entities.columns else ("short_id" if "short_id" in df_entities.columns else df_entities.columns[0])
        col_nome = "title" if "title" in df_entities.columns else ("name" if "name" in df_entities.columns else df_entities.columns[1])
        col_desc = "description" if "description" in df_entities.columns else "summary"
        col_tipo = "type" if "type" in df_entities.columns else None
        
        col_tx_texto = "text" if "text" in df_text_units.columns else "text_unit"
        
        # Filtra e extrai a primeira coluna compatível como uma string válida
        cols_tx_candidatas = [c for c in df_text_units.columns if c in ["entity_ids", "entities", "entity_id"]]
        col_tx_ent = cols_tx_candidatas[0] if cols_tx_candidatas else None

        todas_entidades = []

        # Etapa 1: Processa absolutamente todas as entidades e seus trechos profundos
        for _, entity in df_entities.iterrows():
            ent_id = str(entity[col_ent_id])
            nome = str(entity[col_nome])
            descricao = str(entity[col_desc])
            tipo = str(entity[col_tipo]) if col_tipo else "Conceito de TI"

            if pd.isna(descricao) or str(descricao).strip() == "" or nome.strip() == "":
                continue

            trechos_originais = []
            
            # Garante que col_tx_ent seja tratado como série de texto para evitar erro de DataFrame
            if col_tx_ent and col_tx_texto in df_text_units.columns:
                # Transforma a coluna explicitamente em string na busca
                df_filtrado = df_text_units[df_text_units[col_tx_ent].astype(str).str.contains(f"\\b{re.escape(ent_id)}\\b", na=False)]
                if not df_filtrado.empty:
                    trechos_originais = df_filtrado[col_tx_texto].unique().tolist()

            # Fallback por correspondência direta de texto
            if not trechos_originais and col_tx_texto in df_text_units.columns:
                termo_busca = re.escape(nome)
                df_filtrado = df_text_units[df_text_units[col_tx_texto].astype(str).str.contains(termo_busca, case=False, na=False)]
                if not df_filtrado.empty:
                    trechos_originais = df_filtrado[col_tx_texto].head(4).unique().tolist()

            todas_entidades.append({
                "nome": nome,
                "tipo": tipo,
                "descricao": descricao,
                "trechos": trechos_originais
            })

        total_itens = len(todas_entidades)
        if total_itens == 0:
            print("⚠️ Nenhuma entidade válida encontrada para exportação.")
            return

        # Etapa 2: Calcula a distribuição matemática para enfiar tudo em no máximo 49 arquivos
        num_arquivos_finais = min(MAX_ARQUIVOS, total_itens)
        itens_por_arquivo = math.ceil(total_itens / num_arquivos_finais)

        print(f"📊 Total de conhecimentos: {total_itens}")
        print(f"🗂️ Agrupando tudo em {num_arquivos_finais} arquivos técnicos (aprox. {itens_por_arquivo} conceitos por arquivo).")

        # Etapa 3: Agrupa e escreve os arquivos
        for i in range(num_arquivos_finais):
            inicio = i * itens_por_arquivo
            fim = min(inicio + itens_por_arquivo, total_itens)
            grupo_atual = todas_entidades[inicio:fim]

            if not grupo_atual:
                break

            # Define o nome do arquivo com base no primeiro conceito do bloco atual
            primeiro_conceito = limpar_nome_arquivo(grupo_atual[0]["nome"])
            caminho_arquivo = os.path.join(PASTA_SAIDA, f"BLOCO_{i+1:02d}_{primeiro_conceito}.txt")

            conteudo_bloco = []
            conteudo_bloco.append(f"# 📚 COMPILADO DE TI - BLOCO {i+1} DE {num_arquivos_finais}")
            conteudo_bloco.append(f"Este arquivo agrupa {len(grupo_atual)} conceitos interconectados da base de dados do GraphRAG.\n")
            conteudo_bloco.append("=" * 60 + "\n")

            for item in grupo_atual:
                conteudo_bloco.append(f"## 🧠 CONHECIMENTO: {item['nome'].upper()}")
                conteudo_bloco.append(f"**Categoria Técnica:** {item['tipo']}\n")
                conteudo_bloco.append(f"### 📝 Definição da IA:\n{item['descricao']}\n")
                
                conteudo_bloco.append("### 🔬 Detalhes das Fontes Originais (Trechos Completos dos PDFs):")
                if item["trechos"]:
                    for t_idx, trecho in enumerate(item["trechos"], 1):
                        conteudo_bloco.append(f"\n--- Trecho Técnico {t_idx} ---\n{trecho}")
                else:
                    conteudo_bloco.append("*Nenhum bloco de texto original bruto associado diretamente.*")
                
                conteudo_bloco.append("\n" + "-" * 40 + "\n")

            with open(caminho_arquivo, "w", encoding="utf-8") as f:
                f.write("\n".join(conteudo_bloco))

        print(f"\n🚀 Sucesso! Todo o conteúdo foi exportado sem perdas.")
        print(f"Foram gerados exatamente {num_arquivos_finais} arquivos na pasta '{PASTA_SAIDA}'.")

    except Exception as e:
        print(f"\n❌ Ocorreu um erro no processamento: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    exportar_conteudo_profundo()
