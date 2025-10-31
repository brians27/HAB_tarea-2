
# Tarea 2: Network Propagation

Escribe un script en Python que implemente un ejemplo de propagación en redes utilizando algún algoritmo de **GUILD** y/o **DIAMOnD**. Usa los genes **ENO1**, **PGK1** y **HK2** como semillas. Se proporcionan archivos de red formateada para ambos algoritmos, así como una red filtrada obtenida de STRING y un script de ejemplo para procesarla.

## Estructura del repositorio

```

/network\_propagation/
├── data/
│   ├── network\_guild.txt                        # Red formateada para GUILD
│   ├── network\_diamond.txt                      # Red formateada para DIAMOnD
│   ├── string\_network\_filtered\_hugo-400.tsv     # Red filtrada de STRING
│   └── genes\_seed.txt                           # Genes semilla: ENO1, PGK1, HK2
├── scripts/
│   ├── process\_STRING.py                        # Script de ejemplo para procesar la red
│   └── tu\_script.py                             # Script que debe entregar el estudiante
├── results/                                     # Carpeta para resultados generados
├── README.md                                    # Este archivo
└── requirements.txt                             # Dependencias: networkx, pandas

```

## Instrucciones de entrega

- Haz un **fork** de este repositorio en tu cuenta de GitHub.
- Trabaja en tu fork y sube tu script a la carpeta `scripts/`.
- Tu script debe poder ejecutarse desde la línea de comandos (CLI), aceptando como mínimo el archivo de entrada y el archivo de salida como argumentos.
- Documenta tu código explicando los métodos, librerías y algoritmos utilizados.
- Puedes generar un archivo de resultados en la carpeta `results/` si lo consideras útil.

## Rúbrica de evaluación

La tarea se evaluará sobre un máximo de **10 puntos**, distribuidos según los siguientes criterios:

| Criterio | Descripción | Puntos |
|---------|-------------|--------|
| **1. Funcionalidad** | El script realiza correctamente la propagación en red con GUILD y/o DIAMOnD. | 4 |
| **2. Documentación** | El código está comentado y explica claramente los métodos y algoritmos utilizados. | 2 |
| **3. Uso de librerías** | Se emplean librerías adecuadas para el análisis de redes (e.g., networkx, pandas). | 2 |
| **4. Formato y estilo** | El código sigue buenas prácticas de estilo y es legible. | 1 |
| **5. Automatización (CLI)** | El script acepta argumentos desde la línea de comandos. | 1 |

## Dependencias recomendadas

Incluye en `requirements.txt` las librerías necesarias para ejecutar tu script. Por ejemplo:

```

networkx
pandas

```

