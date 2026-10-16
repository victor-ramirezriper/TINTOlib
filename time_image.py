import glob
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Configuración de rutas base
BASE_RESULTS_DIR = Path(r"C:\EXP_TINTO\Results")
BASE_ANALYSIS_DIR = Path(r"C:\EXP_TINTO\Analysis")

def seleccionar_dataset():
    if not BASE_RESULTS_DIR.exists():
        print(f"[ERROR] No se encontró la carpeta de resultados en: {BASE_RESULTS_DIR}")
        return None

    # Listar todas las carpetas disponibles dentro de Results/
    carpetas_resultados = [f.name for f in BASE_RESULTS_DIR.iterdir() if f.is_dir()]

    if not carpetas_resultados:
        print(f"[ERROR] No hay carpetas de datasets analizados en {BASE_RESULTS_DIR}")
        return None

    print("=" * 60)
    print("DATASETS DISPONIBLES EN RESULTS:")
    print("=" * 60)
    for i, carpeta in enumerate(carpetas_resultados, 1):
        print(f"  {i}. {carpeta}")

    try:
        opcion = int(input("\nSelecciona el número del dataset a graficar: ")) - 1
        if 0 <= opcion < len(carpetas_resultados):
            return carpetas_resultados[opcion]
        else:
            print("[ERROR] Número fuera de rango.")
            return None
    except ValueError:
        print("[ERROR] Por favor, ingresa un número válido.")
        return None

def analizar_recursos_dataset(nombre_carpeta_dataset):
    dataset_path = BASE_RESULTS_DIR / nombre_carpeta_dataset
    
    # Modelos actualizados incluyendo ViT
    modelos = ["EfficientNet-B0", "ResNet18", "ViT"]
    
    # Directorio de salida personalizado para este dataset
    nombre_limpio = nombre_carpeta_dataset.replace("_Original", "")
    output_dir = BASE_ANALYSIS_DIR / nombre_limpio
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"PROCESANDO ANÁLISIS DE RECURSOS PARA: {nombre_carpeta_dataset}")
    print(f"Directorio de salida: {output_dir}")
    print(f"{'='*60}\n")

    for modelo in modelos:
        # Buscar carpetas del modelo que contengan el timestamp dinámico
        patron = str(dataset_path / f"{modelo}_*")
        coincidencias = glob.glob(patron)
        
        if not coincidencias:
            print(f"[ADVERTENCIA] No se encontraron carpetas de resultados para el modelo: {modelo}")
            continue
            
        # Tomar la ejecución más reciente en caso de haber varias
        modelo_path = Path(sorted(coincidencias)[0])
        archivo_folds = modelo_path / "resultados_folds.csv"
        
        if not archivo_folds.exists():
            print(f"[ADVERTENCIA] El archivo 'resultados_folds.csv' no existe en: {modelo_path}")
            continue
            
        print(f"-> Generando gráficas de hardware para el modelo: {modelo}...")
        df_folds = pd.read_csv(archivo_folds)
        
        # Configuración de estilo visual
        sns.set_theme(style="whitegrid")
        
        # 1. Gráfica de Tiempos de Entrenamiento por Método y Fold
        plt.figure(figsize=(14, 7))
        sns.barplot(data=df_folds, x="Metodo", y="Tiempo_Entrenamiento_s", hue="Fold", palette="viridis")
        plt.title(f"Tiempo de Entrenamiento por Método y Fold\nModelo: {modelo} | Dataset: {nombre_limpio}", fontsize=12, fontweight="bold")
        plt.xlabel("Método de Transformación", fontsize=10)
        plt.ylabel("Tiempo de Entrenamiento (Segundos)", fontsize=10)
        plt.xticks(rotation=45, ha="right")
        plt.legend(title="Fold", bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(output_dir / f"{modelo}_tiempo_entrenamiento.png", dpi=300)
        plt.close()
        
        # 2. Gráfica de Consumo Máximo de RAM (GB)
        plt.figure(figsize=(14, 7))
        sns.boxplot(data=df_folds, x="Metodo", y="Max_RAM_GB", palette="Set2")
        plt.title(f"Distribución del Consumo Máximo de RAM (GB) por Método\nModelo: {modelo} | Dataset: {nombre_limpio}", fontsize=12, fontweight="bold")
        plt.xlabel("Método de Transformación", fontsize=10)
        plt.ylabel("RAM Máxima Consumida (GB)", fontsize=10)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(output_dir / f"{modelo}_consumo_max_ram.png", dpi=300)
        plt.close()
        
        # 3. Gráfica de Uso Promedio de CPU (%)
        plt.figure(figsize=(14, 7))
        sns.barplot(data=df_folds, x="Metodo", y="Avg_CPU_Percent", palette="magma", errorbar=None)
        plt.title(f"Uso Promedio de CPU (%) por Método (Promedio de Folds)\nModelo: {modelo} | Dataset: {nombre_limpio}", fontsize=12, fontweight="bold")
        plt.xlabel("Método de Transformación", fontsize=10)
        plt.ylabel("Uso Promedio de CPU (%)", fontsize=10)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(output_dir / f"{modelo}_uso_promedio_cpu.png", dpi=300)
        plt.close()

    print(f"\n{'='*60}")
    print("¡PROCESO FINALIZADO CON ÉXITO!")
    print(f"Todas las gráficas de recursos se guardaron en: {output_dir}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    carpeta_seleccionada = seleccionar_dataset()
    if carpeta_seleccionada:
        analizar_recursos_dataset(carpeta_seleccionada)