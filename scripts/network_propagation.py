"""
============================================================
WORKFLOW OF THE UNIFIED SCRIPT

1. STRING NETWORK FILTERING AND CONVERSION:
   - Reads the STRING network file (protein links) with combined scores.
   - Filters only interactions with combined_score >= 800.
   - Converts ENSP IDs to Entrez Gene IDs using MyGene.

2. SEED GENES READING:
   - Reads seed genes from the 'genes_seed.txt' file.
   - Converts any HUGO symbols to Entrez Gene IDs.

3. GRAPH CONSTRUCTION:
   - Builds a NetworkX graph with nodes representing genes and edges representing STRING interactions.
   - Edge weights correspond to the combined_score in the filtered network.

4. CHECK SEED GENE PRESENCE:
   - Checks which seed genes are present in the network.
   - If none of the seed genes are present, the script stops.

5. ALGORITHMS EXECUTION:
   a) DIAMOnD:
      - Adds nodes to the network based on statistical proximity to seed genes using the hypergeometric test.
      - Saves the results in 'diamond_results.csv'.
   b) GUILD NetScore:
      - Calculates a score for all nodes by propagating information from seed genes to neighbors.
      - Saves the results in 'guild_results.csv'.

6. OUTPUT:
   - All output files are saved in the 'results' folder.
   - Node IDs in the results are in Entrez Gene ID format.
============================================================
"""

import os
import pandas as pd
import mygene
from tqdm import tqdm
import networkx as nx
from scipy.stats import hypergeom

# ==============================
# PATH CONFIGURATION
# ==============================
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, '..', 'data')
results_dir = os.path.join(base_dir, '..', 'results')

if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Input files
string_raw_file = os.path.join(data_dir, '9606.protein.links.v12.0.txt')
genes_seed_file = os.path.join(data_dir, 'genes_seed.txt')
string_entrez_file = os.path.join(results_dir, 'string12_9606_800_entrez.txt')

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
print("Seed genes read from 'genes_seed.txt':", seed_genes)

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

print(f"\nSeed genes present in the network: {len(present)}, missing: {len(missing)}")
if missing:
    print("Missing:", missing)
if len(present) == 0:
    raise ValueError("None of the seed genes are present in the network!")

# ==============================
# ALGORITHM EXECUTION
# ==============================
diamond_results = diamond_algorithm(G_string, present, num_nodes=10)
diamond_out = pd.DataFrame(diamond_results, columns=['node', 'p_value'])
diamond_out.to_csv(os.path.join(results_dir, 'diamond_results.csv'), index=False)

guild_scores = guild_netscore(G_string, present)
guild_out = pd.DataFrame(sorted(guild_scores.items(), key=lambda x: x[1], reverse=True),
                         columns=['node', 'score'])
guild_out.to_csv(os.path.join(results_dir, 'guild_results.csv'), index=False)

top10 = guild_out.head(10)
print("\nTop 10 GUILD NetScore nodes:")
for _, row in top10.iterrows():
    print(f"{row['node']} {row['score']:.3f}")

