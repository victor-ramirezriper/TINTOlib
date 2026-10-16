import os
import time
from pathlib import Path
from datetime import datetime
import pandas as pd
import json

from src.project_config import IMAGE_DIR, RESULTS_DIR
from deep_learning_manager import DeepLearningPipelineManager, fijar_semillas

def menu_interactivo_evaluacion():
    print("=" * 60)
    print("EVALUACIÓN DE DEEP LEARNING (DATASETS GRANDES - HOLDOUT / 1 FOLD)")
    print("=" * 60)

    carpeta_experimentos = Path(IMAGE_DIR) / "Experimentos"
    if not carpeta_experimentos.exists():
        print("\nNo se encontró la carpeta de Experimentos en Image/")
        exit()

    # Filtrar solo lotes que correspondan a gran escala o listar todos
    lotes = sorted([d.name for d in carpeta_experimentos.iterdir() if d.is_dir()])
    if not lotes:
        print("\nNo hay lotes de experimentos disponibles.")
        exit()
        
    print("\nLotes de experimentos disponibles:")
    for i, lote in enumerate(lotes, 1):
        print(f"  {i}. {lote}")
    
    opcion_lote = int(input("\nSelecciona el número del lote a evaluar: ")) - 1
    lote_seleccionado = lotes[opcion_lote]
    ruta_lote = carpeta_experimentos / lote_seleccionado

    # Buscar las carpetas de métodos TRAIN dentro del lote
    # (Pueden estar directamente o dentro de una subcarpeta 'Imagenes')
    ruta_imagenes = ruta_lote / "Imagenes" if (ruta_lote / "Imagenes").exists() else ruta_lote
    
    metodos_train = sorted([
        d.name.replace("EXP_", "").replace("_TRAIN", "")
        for d in ruta_imagenes.iterdir() 
        if d.is_dir() and d.name.endswith("_TRAIN")
    ])

    if not metodos_train:
        print(f"\nNo se encontraron métodos TRAIN en {ruta_lote}")
        exit()

    print("\nMétodos generados encontrados en este lote:")
    for i, metodo in enumerate(metodos_train, 1):
        print(f"  {i}. {metodo}")
    print(f"  {len(metodos_train) + 1}. TODOS LOS MÉTODOS")

    opcion_met = int(input("\nSelecciona el número del método a evaluar: "))
    
    if opcion_met == len(metodos_train) + 1:
        metodos_a_evaluar = metodos_train
    else:
        metodos_a_evaluar = [metodos_train[opcion_met - 1]]

    redes_disponibles = ["ResNet18", "EfficientNet-B0", "ViT-Tiny"]
    print("\nArquitecturas de Redes Neuronales:")
    for i, red in enumerate(redes_disponibles, 1):
        print(f"  {i}. {red}")
        
    opcion_red = int(input("\nSelecciona el número de la arquitectura: ")) - 1
    red_seleccionada = redes_disponibles[opcion_red]

    return lote_seleccionado, ruta_imagenes, metodos_a_evaluar, red_seleccionada


if __name__ == "__main__":
    fijar_semillas(42)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    lote, ruta_imagenes, metodos_a_evaluar, red_seleccionada = menu_interactivo_evaluacion()

    print("\n" + "=" * 60)
    print(f"INICIANDO ENTRENAMIENTO Y EVALUACIÓN (GRAN ESCALA)")
    print(f"Lote     : {lote}")
    print(f"Red      : {red_seleccionada}")
    print(f"Métodos  : {metodos_a_evaluar}")
    print("=" * 60)

    # Instanciar el manager de deep learning
    manager = DeepLearningPipelineManager(
        red_seleccionada=red_seleccionada,
        epochs=10,         # Puedes ajustar las épocas aquí si lo deseas
        batch_size=32,
        lr=0.001
    )

    resultados_totales = []

    for metodo in metodos_a_evaluar:
        try:
            print(f"\n--- Evaluando método: {metodo.upper()} ---")
            res = manager.entrenar_holdout_gran_escala(ruta_imagenes, metodo)
            resultados_totales.append(res)
        except Exception as e:
            print(f"[!] Error crítico evaluando {metodo}: {e}")

    # Guardar resultados globales en Results/
    if resultados_totales:
        nombre_dataset = lote.split("_LargeScale")[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        carpeta_resultados = RESULTS_DIR / "Experimentos_Grandes" / f"{nombre_dataset}_{red_seleccionada}_{timestamp}"
        carpeta_resultados.mkdir(parents=True, exist_ok=True)

        df_resumen = pd.DataFrame(resultados_totales)
        ruta_csv = carpeta_resultados / "resumen_evaluacion_holdout.csv"
        df_resumen.to_csv(ruta_csv, index=False)

        print("\n" + "=" * 60)
        print("¡EVALUACIÓN DE GRAN ESCALA FINALIZADA CON ÉXITO!")
        print(f"Resultados guardados en: {ruta_csv}")
        print("=" * 60)
        print(df_resumen.to_string(index=False))