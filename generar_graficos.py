import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configuración de rutas
RESULTS_DIR = Path("Results")
ANALYSIS_DIR = Path("Analysis")

def seleccionar_dataset_analisis():
    print("=" * 60)
    print("MÓDULO DE ANÁLISIS VISUAL Y GRAFICACIÓN")
    print("=" * 60)
    
    # Buscar datasets que ya tienen un Baseline evaluado
    archivos_base = list(RESULTS_DIR.glob("Baseline_*.csv"))
    if not archivos_base:
        print("No se encontraron resultados de Línea Base en la carpeta Results/.")
        exit()
        
    datasets_disponibles = [f.stem.replace("Baseline_", "") for f in archivos_base]
    
    print("\nDatasets con resultados disponibles:")
    for i, ds in enumerate(datasets_disponibles, 1):
        print(f"  {i}. {ds}")
        
    opcion = int(input("\nSelecciona el número del dataset a graficar: ")) - 1
    return datasets_disponibles[opcion]

def unificar_datos(dataset_nombre):
    df_lista = []
    
    # 1. Cargar Línea Base (Vectores)
    ruta_base = RESULTS_DIR / f"Baseline_{dataset_nombre}.csv"
    if ruta_base.exists():
        df_base = pd.read_csv(ruta_base)
        # Adaptar columnas para unificarlas con las de Deep Learning
        df_base = df_base.rename(columns={"Modelo": "Metodo_o_Modelo"})
        df_base["Arquitectura"] = "Línea Base (Tabular)"
        df_lista.append(df_base[["Arquitectura", "Metodo_o_Modelo", "Accuracy", "F1_Score", "ROC_AUC", "Tiempo_s"]])

    # 2. Cargar Resultados de Deep Learning (CNN / ViT)
    ruta_dl_dir = RESULTS_DIR / dataset_nombre
    if ruta_dl_dir.exists():
        for archivo_csv in ruta_dl_dir.glob("*_Results.csv"):
            df_dl = pd.read_csv(archivo_csv)
            df_dl = df_dl.rename(columns={
                "Metodo_TINTO": "Metodo_o_Modelo",
                "Modelo_Red": "Arquitectura",
                "Tiempo_Prom_Fold_s": "Tiempo_s"
            })
            df_lista.append(df_dl[["Arquitectura", "Metodo_o_Modelo", "Accuracy", "F1_Score", "ROC_AUC", "Tiempo_s"]])

    if not df_lista:
        print(f"No se encontraron datos para {dataset_nombre}")
        exit()

    return pd.concat(df_lista, ignore_index=True)

def generar_y_guardar_graficos(df_unificado, dataset_nombre):
    # Crear carpeta de Análisis específica para este dataset
    carpeta_salida = ANALYSIS_DIR / dataset_nombre
    carpeta_salida.mkdir(parents=True, exist_ok=True)
    
    # Configurar estilo de Seaborn para gráficos académicos
    sns.set_theme(style="whitegrid")
    
    metricas = {
        "Accuracy": "Precisión Global (Accuracy %)",
        "F1_Score": "F1-Score Macro (%)",
        "ROC_AUC": "Área bajo la curva (ROC-AUC %)",
        "Tiempo_s": "Tiempo de Procesamiento (Segundos)"
    }
    
    colores = sns.color_palette("Set2")

    print(f"\nGenerando gráficos para: {dataset_nombre}...")
    
    for columna, titulo_eje in metricas.items():
        # Limpiar valores nulos (por si un baseline binario viejo tiene NaN en ROC-AUC)
        df_limpio = df_unificado.dropna(subset=[columna])
        
        plt.figure(figsize=(14, 7))
        
        # Crear gráfico de barras agrupadas
        grafico = sns.barplot(
            data=df_limpio, 
            x="Metodo_o_Modelo", 
            y=columna, 
            hue="Arquitectura",
            palette=colores
        )
        
        # Personalización de la gráfica
        plt.title(f"Comparativa de {titulo_eje}\nDataset: {dataset_nombre}", fontsize=14, pad=15)
        plt.xlabel("Método Tabular / Algoritmo TINTOlib", fontsize=12)
        plt.ylabel(titulo_eje, fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.legend(title="Arquitectura de Red", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # Guardar como imagen PNG
        nombre_archivo = f"Comparativa_{columna}.png"
        ruta_imagen = carpeta_salida / nombre_archivo
        plt.savefig(ruta_imagen, dpi=300, bbox_inches='tight')
        plt.close() # Cierra la figura en memoria sin mostrarla en pantalla
        
        print(f"  -> Guardado: {ruta_imagen}")

if __name__ == "__main__":
    dataset = seleccionar_dataset_analisis()
    df_total = unificar_datos(dataset)
    generar_y_guardar_graficos(df_total, dataset)
    print("\nAnálisis visual completado con éxito.")