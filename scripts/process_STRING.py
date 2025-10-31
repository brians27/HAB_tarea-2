import pandas as pd
import mygene
from tqdm import tqdm
import os


# Inicializar MyGene
mg = mygene.MyGeneInfo()

# Ruta al archivo de entrada y salida
input_file = '9606.protein.links.v12.0.txt'  # Reemplaza con la ruta a tu archivo
output_file = 'string12_9606_800_entrez.txt'

# Definir el tamaño de los chunks
chunk_size = 1000000  # Ajusta según la memoria disponible

# Inicializar conjuntos para almacenar IDs únicos
unique_ids = set()

# Primero, filtrar y recopilar IDs únicos
print("Filtrando y recopilando IDs únicos...")
filtered_chunks = []
for chunk in tqdm(pd.read_csv(input_file, sep=' ', header=0, chunksize=chunk_size)):
    # Filtrar por combined_score >= 800
    filtered = chunk[chunk['combined_score'] >= 800]
    filtered_chunks.append(filtered)
    
    # Extraer IDs únicos de protein1 y protein2
    unique_ids.update(filtered['protein1'].unique())
    unique_ids.update(filtered['protein2'].unique())

# Concatenar todos los chunks filtrados
filtered_data = pd.concat(filtered_chunks, ignore_index=True)
del filtered_chunks  # Liberar memoria

# Extraer los IDs ENSP (remover el prefijo '9606.')
ensp_ids = [id_.split('.')[1] for id_ in unique_ids]

# Mapear ENSP a ENTREZ usando MyGene
print("Mapeando ENSP a ENTREZ...")
batch_size = 1000  # Tamaño de batch para las consultas
entrez_mapping = {}
for i in tqdm(range(0, len(ensp_ids), batch_size)):
    batch = ensp_ids[i:i+batch_size]
    # Buscar los IDs ENSP
    results = mg.querymany(batch, scopes='ensembl.protein', fields='entrezgene', species='human')
    for res in results:
        ensp = res.get('query')
        entrez = res.get('entrezgene')
        if entrez:
            entrez_mapping[f'9606.{res["query"]}'] = entrez

# Función para mapear IDs
def map_to_entrez(id_):
    return entrez_mapping.get(id_, None)

# Aplicar el mapeo a las columnas de proteínas
print("Aplicando el mapeo a las columnas de proteínas...")
filtered_data['protein1_entrez'] = filtered_data['protein1'].apply(map_to_entrez)
filtered_data['protein2_entrez'] = filtered_data['protein2'].apply(map_to_entrez)

# Opcional: eliminar filas donde no se encontró el mapeo
filtered_data.dropna(subset=['protein1_entrez', 'protein2_entrez'], inplace=True)

# Seleccionar las columnas deseadas
final_data = filtered_data[['protein1_entrez', 'protein2_entrez', 'combined_score']]

# Guardar el resultado
print(f"Guardando los datos filtrados y mapeados en {output_file}...")
final_data.to_csv(output_file, sep='\t', index=False)

print("Proceso completado exitosamente.")

