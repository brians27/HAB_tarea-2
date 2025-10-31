import os
import pandas as pd
import networkx as nx
from scipy.stats import hypergeom
import mygene

# ==============================
# CONFIGURAZIONE PERCORSI
# ==============================
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, '..', 'data')
results_dir = os.path.join(base_dir, '..', 'results')

# File input
string_file = os.path.join(results_dir, 'string12_9606_800_entrez.txt')
genes_seed_file = os.path.join(data_dir, 'genes_seed.txt')

# Assicurati che la cartella results esista
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# ==============================
# LETTURA DEI GENI SEME
# ==============================
with open(genes_seed_file, 'r') as f:
    genes_semi = [g.strip() for g in f.read().splitlines()]

print("Geni letti da 'genes_seed.txt':", genes_semi)

# Conversione in Entrez ID usando MyGene (solo se ancora non li hai in Entrez)
mg = mygene.MyGeneInfo()
res = mg.querymany(genes_semi, scopes='symbol', fields='entrezgene', species='human')
semi_entrez = [str(r['entrezgene']) for r in res if 'entrezgene' in r]
print("Geni semi convertiti in Entrez ID:", semi_entrez)

# ==============================
# LETTURA DELLA RETE STRING
# ==============================
def leggi_rete_string(file):
    """
    Legge la rete STRING filtrata già con Entrez ID.
    """
    df = pd.read_csv(file, sep='\t')
    G = nx.Graph()
    for _, row in df.iterrows():
        G.add_edge(str(row['protein1_entrez']), str(row['protein2_entrez']), weight=row['combined_score'])
    return G

G_string = leggi_rete_string(string_file)
print(f"Rete STRING caricata con {G_string.number_of_nodes()} nodi e {G_string.number_of_edges()} archi.")

# ==============================
# FUNZIONI DI ALGORITMI
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
        print(f"DIAMOnD: aggiunto nodo {best_node} p-value {best_p:.2e}")
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
            if not neighbors:
                new_score[node] = score[node]
            else:
                new_score[node] = sum(score[n] for n in neighbors) / len(neighbors)
        score = new_score
    return score

# ==============================
# VERIFICA PRESENZA GENI SEME NELLA RETE
# ==============================
presenti = [g for g in semi_entrez if g in G_string.nodes()]
mancanti = [g for g in semi_entrez if g not in G_string.nodes()]

print("\n--- Verifica presenza geni semi ---")
print(f"Presenti: {len(presenti)}, Mancanti: {len(mancanti)}")
if mancanti:
    print("❌ Mancanti:", mancanti)

if len(presenti) == 0:
    raise ValueError("⚠️ Nessuno dei geni semi è presente nella rete STRING! Controlla i formati o le conversioni.")

# ==============================
# ESECUZIONE DEGLI ALGORITMI
# ==============================
print("\n--- DIAMOnD ---")
diamond_results = diamond_algorithm(G_string, presenti, num_nodes=10)
diamond_out = pd.DataFrame(diamond_results, columns=['node', 'p_value'])
diamond_out.to_csv(os.path.join(results_dir, 'diamond_results.csv'), index=False)
print(f"Risultati DIAMOnD salvati in '{os.path.join(results_dir, 'diamond_results.csv')}'")

print("\n--- GUILD NetScore ---")
guild_scores = guild_netscore(G_string, presenti)
guild_out = pd.DataFrame(sorted(guild_scores.items(), key=lambda x: x[1], reverse=True),
                         columns=['node', 'score'])
guild_out.to_csv(os.path.join(results_dir, 'guild_results.csv'), index=False)
print(f"Risultati GUILD salvati in '{os.path.join(results_dir, 'guild_results.csv')}'")

top10 = guild_out.head(10)
print("\nTop 10 nodi GUILD NetScore:")
for _, row in top10.iterrows():
    print(f"{row['node']} {row['score']:.3f}")
