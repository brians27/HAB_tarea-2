#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================
WORKFLOW OF THE UNIFIED SCRIPT

1. STRING NETWORK FILTERING AND CONVERSION:
   - Reads the STRING network file (protein links) with combined scores.
   - Filters only interactions with combined_score >= 800.
   - Converts ENSP IDs to Entrez Gene IDs using MyGene.

2. SEED GENES READING:
   - Reads seed genes from the file.
   - Converts HUGO symbols to Entrez Gene IDs.

3. GRAPH CONSTRUCTION:
   - Builds a NetworkX graph with nodes representing genes and edges representing STRING interactions.
   - Edge weights correspond to the combined_score in the filtered network.

4. CHECK SEED GENE PRESENCE:
   - Checks which seed genes are present in the network.

5. ALGORITHMS EXECUTION:
   a) DIAMOnD:
      - Adds nodes to the network based on proximity to seed genes using hypergeometric test.
   b) GUILD NetScore:
      - Calculates a score for all nodes propagating information from seed genes.

6. SELECT TOP NODES:
   - Selects top nodes from DIAMOnD and GUILD for functional analysis.

7. FUNCTIONAL ENRICHMENT:
   - Converts Entrez IDs to HUGO symbols.
   - Uses Enrichr API to perform ORA on top nodes.
   - Generates bar plots for top terms.

8. OUTPUT:
   - All output files and plots are saved in the output folder.
============================================================
"""

import os
import argparse
import pandas as pd
import mygene
import networkx as nx
from scipy.stats import hypergeom
from tqdm import tqdm
import requests
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================
# CLI ARGUMENTS
# ==============================
parser = argparse.ArgumentParser(description="STRING network + DIAMOnD + GUILD + Functional Analysis")
parser.add_argument("--string_input", type=str, required=True, help="Path to raw STRING network file")
parser.add_argument("--seed_genes", type=str, required=True, help="Path to seed genes file")
parser.add_argument("--output_dir", type=str, default="../results", help="Folder to save results")
parser.add_argument("--top_n", type=int, default=10, help="Number of top nodes to select for functional analysis")
args = parser.parse_args()

string_raw_file = args.string_input
genes_seed_file = args.seed_genes
results_dir = args.output_dir
TOP_N = args.top_n
os.makedirs(results_dir, exist_ok=True)

string_entrez_file = os.path.join(results_dir, 'string_filtered_entrez.txt')

# ==============================
# STRING FILTERING AND ENTREZ CONVERSION
# ==============================
mg = mygene.MyGeneInfo()
chunk_size = 1000000
unique_ids = set()
filtered_chunks = []

print("Filtering STRING network (score >= 800)...")
for chunk in tqdm(pd.read_csv(string_raw_file, sep=' ', header=0, chunksize=chunk_size)):
    filtered = chunk[chunk['combined_score'] >= 800]
    filtered_chunks.append(filtered)
    unique_ids.update(filtered['protein1'].unique())
    unique_ids.update(filtered['protein2'].unique())

filtered_data = pd.concat(filtered_chunks, ignore_index=True)
del filtered_chunks

ensp_ids = [id_.split('.')[1] for id_ in unique_ids]

print("Mapping ENSP to ENTREZ...")
entrez_mapping = {}
batch_size = 1000
for i in tqdm(range(0, len(ensp_ids), batch_size)):
    batch = ensp_ids[i:i+batch_size]
    results = mg.querymany(batch, scopes='ensembl.protein', fields='entrezgene', species='human')
    for res in results:
        if 'entrezgene' in res:
            entrez_mapping[f'9606.{res["query"]}'] = res['entrezgene']

def map_to_entrez(id_):
    return entrez_mapping.get(id_, None)

filtered_data['protein1_entrez'] = filtered_data['protein1'].apply(map_to_entrez)
filtered_data['protein2_entrez'] = filtered_data['protein2'].apply(map_to_entrez)
filtered_data.dropna(subset=['protein1_entrez', 'protein2_entrez'], inplace=True)

final_data = filtered_data[['protein1_entrez', 'protein2_entrez', 'combined_score']]
final_data.to_csv(string_entrez_file, sep='\t', index=False)
print(f"Filtered STRING network saved to '{string_entrez_file}'")

# ==============================
# SEED GENES READING
# ==============================
with open(genes_seed_file, 'r') as f:
    seed_genes = [g.strip() for g in f.read().splitlines()]
print("Seed genes read:", seed_genes)

res = mg.querymany(seed_genes, scopes='symbol', fields='entrezgene', species='human')
seed_entrez = [str(r['entrezgene']) for r in res if 'entrezgene' in r]
print("Seed genes converted to Entrez IDs:", seed_entrez)

# ==============================
# NETWORK CONSTRUCTION
# ==============================
def read_string_network(file):
    df = pd.read_csv(file, sep='\t')
    G = nx.Graph()
    for _, row in df.iterrows():
        G.add_edge(str(row['protein1_entrez']), str(row['protein2_entrez']),
                   weight=row['combined_score'])
    return G

G_string = read_string_network(string_entrez_file)
print(f"STRING network loaded with {G_string.number_of_nodes()} nodes and {G_string.number_of_edges()} edges.")

# ==============================
# ALGORITHM FUNCTIONS
# ==============================
def diamond_algorithm(G, seed_genes, num_nodes=10):
    """
    DIAMOnD algorithm: iteratively adds nodes statistically close to seed genes using hypergeometric test.
    """
    added_nodes = []
    all_nodes = set(G.nodes())
    seed_set = set(seed_genes)
    candidate_nodes = all_nodes - seed_set

    for _ in range(num_nodes):
        best_node, best_p = None, 1
        for node in candidate_nodes:
            neighbors = set(G.neighbors(node))
            k_s = len(neighbors & seed_set)
            k = len(neighbors)
            K = len(seed_set)
            N = len(G)
            p = hypergeom.sf(k_s - 1, N, K, k)
            if p < best_p:
                best_node, best_p = node, p
        if best_node is None:
            break
        added_nodes.append((best_node, best_p))
        seed_set.add(best_node)
        candidate_nodes.remove(best_node)
    return added_nodes

def guild_netscore(G, seed_genes, max_iter=5):
    """
    GUILD NetScore: propagates scores from seed genes through neighbors to rank all nodes.
    """
    score = {n: 0.0 for n in G.nodes()}
    for g in seed_genes:
        if g in score:
            score[g] = 1.0
    for _ in range(max_iter):
        new_score = {}
        for node in G.nodes():
            neighbors = list(G.neighbors(node))
            if neighbors:
                new_score[node] = sum(score[n] for n in neighbors) / len(neighbors)
            else:
                new_score[node] = score[node]
        score = new_score
    return score

# ==============================
# CHECK SEED GENE PRESENCE
# ==============================
present = [g for g in seed_entrez if g in G_string.nodes()]
missing = [g for g in seed_entrez if g not in G_string.nodes()]
print(f"\nSeed genes present: {len(present)}, missing: {len(missing)}")
if missing:
    print("Missing:", missing)
if len(present) == 0:
    raise ValueError("None of the seed genes are present in the network!")

# ==============================
# ALGORITHM EXECUTION
# ==============================
diamond_results = diamond_algorithm(G_string, present, num_nodes=TOP_N)
diamond_out_file = os.path.join(results_dir, 'diamond_results.csv')
pd.DataFrame(diamond_results, columns=['node', 'p_value']).to_csv(diamond_out_file, index=False)
print(f"DIAMOnD results saved to '{diamond_out_file}'")

guild_scores = guild_netscore(G_string, present)
guild_out_file = os.path.join(results_dir, 'guild_results.csv')
pd.DataFrame(sorted(guild_scores.items(), key=lambda x: x[1], reverse=True),
             columns=['node', 'score']).to_csv(guild_out_file, index=False)
print(f"GUILD NetScore results saved to '{guild_out_file}'")

# ==============================
# SELECT TOP NODES FOR FUNCTIONAL ANALYSIS
# ==============================
diamond_df = pd.read_csv(diamond_out_file)
top_diamond_nodes = diamond_df.nsmallest(TOP_N, 'p_value')['node'].astype(str).tolist()

guild_df = pd.read_csv(guild_out_file)
top_guild_nodes = guild_df.nlargest(TOP_N, 'score')['node'].astype(str).tolist()

overlap_nodes = list(set(top_diamond_nodes) & set(top_guild_nodes))
print(f"\nOverlap top {TOP_N} nodes DIAMOnD vs GUILD: {len(overlap_nodes)}")
if overlap_nodes:
    print(overlap_nodes)

# Combine top nodes
nodes_for_functional = list(set(top_diamond_nodes + top_guild_nodes))

# ==============================
# MAP ENTIRE LIST OF NODES TO HUGO SYMBOLS FOR FUNCTIONAL ANALYSIS
# ==============================
def entrez_to_symbol(entrez_ids: list[str]) -> list[str]:
    """
    Convert a list of Entrez IDs to HUGO symbols using MyGene.
    Only returns genes that are successfully mapped.
    """
    batch_size = 1000
    symbols = []
    for i in range(0, len(entrez_ids), batch_size):
        batch = entrez_ids[i:i+batch_size]
        results = mg.getgenes(batch, fields='symbol', species='human')
        for res in results:
            if 'symbol' in res:
                symbols.append(res['symbol'])
    return symbols

nodes_for_functional_symbols = entrez_to_symbol(nodes_for_functional)
print(f"\nMapped {len(nodes_for_functional_symbols)} nodes to HUGO symbols for functional enrichment.")

functional_input_file = os.path.join(results_dir, "nodes_for_functional.txt")
with open(functional_input_file, "w") as f:
    for gene in nodes_for_functional_symbols:
        f.write(f"{gene}\n")
print(f"Nodes for functional analysis saved to '{functional_input_file}'")


# ==============================
# FUNCTIONAL ENRICHMENT
# ==============================
def enrichr_analysis(genes: list[str], libraries: list[str]) -> dict:
    """
    Submit genes to Enrichr API and fetch top enriched terms.
    """
    add_list_url = "https://maayanlab.cloud/Enrichr/addList"
    enrich_url = "https://maayanlab.cloud/Enrichr/enrich"
    gene_str = "\n".join(genes)
    payload = {'list': (None, gene_str), 'description': (None, 'Functional analysis')}
    response = requests.post(add_list_url, files=payload)
    if response.status_code != 200:
        raise RuntimeError("Error submitting gene list to Enrichr.")
    user_list_id = response.json()['userListId']
    enrichment_results = {}
    for lib in libraries:
        params = {'userListId': user_list_id, 'backgroundType': lib}
        enrich_response = requests.get(enrich_url, params=params)
        if enrich_response.status_code != 200:
            print(f"Error retrieving enrichment results for {lib}")
            continue
        results = enrich_response.json()
        if lib not in results:
            continue
        enrichment_results[lib] = results[lib][:5]
    return enrichment_results

def plot_enrichment(enrichment_results: dict, out_dir: str) -> None:
    """
    Create barplots for enrichment results.
    """
    sns.set(style="whitegrid")
    for lib, terms_data in enrichment_results.items():
        if not terms_data:
            continue
        terms = [entry[1] for entry in terms_data]
        scores = [entry[4] for entry in terms_data]
        plt.figure(figsize=(8, 5))
        sns.barplot(x=scores, y=terms, palette="viridis")
        plt.xlabel("Combined Score")
        plt.ylabel("Term")
        plt.title(f"Top 5 Enriched Terms: {lib}")
        plt.tight_layout()
        plot_filename = f"top5_{lib.replace(' ', '_')}.png"
        plot_path = os.path.join(out_dir, plot_filename)
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Saved barplot for {lib} to '{plot_path}'")

print("\nPerforming functional enrichment on top nodes...")
libraries = [
    'KEGG_2021_Human',
    'GO_Biological_Process_2023',
    'GO_Molecular_Function_2023',
    'GO_Cellular_Component_2023'
]
functional_results = enrichr_analysis(nodes_for_functional_symbols, libraries)
plot_enrichment(functional_results, results_dir)
print("Functional analysis completed successfully.")
