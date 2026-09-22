import os
import time
import pandas as pd
import numpy as np
import warnings
from datetime import datetime

# NUEVO IMPORT: Para registrar el hardware
import psutil 

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from imblearn.over_sampling import SMOTE 

from src.project_config import obtener_ruta_dataset, preparar_directorios, DATA_DIR, RESULTS_DIR
from src.dataset_analyzer import cargar_dataset
from src.preprocessing import (
    limpiar_dataset, 
    aplicar_encoding_seguro, 
    aplicar_escalado_seguro,
    codificar_target_seguro
)

warnings.filterwarnings('ignore')

def menu_interactivo():
    print("=" * 60)
    print("LÍNEA BASE TABULAR (VECTORES) - SCIKIT-LEARN")
    print("=" * 60)

    archivos_csv = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    if not archivos_csv:
        print("\nNo se encontraron archivos .csv en la carpeta Data/")
        exit()
        
    print("\nDatasets disponibles:")
    for i, archivo in enumerate(archivos_csv, 1):
        print(f"  {i}. {archivo}")
    
    opcion_ds = int(input("\nSelecciona el número del dataset: ")) - 1
    nombre_archivo = archivos_csv[opcion_ds]
    ruta_dataset = obtener_ruta_dataset(nombre_archivo)
    
    df_temp = pd.read_csv(ruta_dataset, nrows=0) 
    columnas = df_temp.columns.tolist()
    
    print(f"\nColumnas en '{nombre_archivo}':")
    for i, col in enumerate(columnas, 1):
        print(f"  {i}. {col}")
        
    opcion_tg = int(input("\nSelecciona el número del Target: ")) - 1
    target_col = columnas[opcion_tg]

    return nombre_archivo, ruta_dataset, target_col

def evaluar_modelos_tabular(X, y, columnas_categoricas, columnas_numericas, nombre_dataset):
    conteo_clases = y.value_counts()
    clase_mayoritaria = conteo_clases.max()
    clase_minoritaria = conteo_clases.min()
    ratio = clase_mayoritaria / clase_minoritaria if clase_minoritaria > 0 else float('inf')
    
    filas_csv = []
    
    # Inicializar el monitor de procesos de hardware
    proceso = psutil.Process(os.getpid())
    
    estrategias = ["Original"]
    if ratio > 2.0:
        print(f"\nADVERTENCIA: Alto desbalanceo detectado (Ratio {ratio:.2f}).")
        print("MODO AUTOMÁTICO: Se ejecutarán y guardarán AMBOS análisis (Original y SMOTE).")
        estrategias.append("SMOTE")
    else:
        print(f"\nDataset balanceado (Ratio {ratio:.2f}). Solo se ejecutará la versión Original.")

    for estrategia in estrategias:
        print(f"\n" + "="*85)
        print(f"INICIANDO EVALUACIÓN: ESTRATEGIA {estrategia.upper()}")
        print("="*85)
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        modelos = {
            "Random_Forest": RandomForestClassifier(random_state=42),
            "SVM": SVC(probability=True, random_state=42),
            "MLP_Clasico": MLPClassifier(random_state=42, max_iter=1000)
        }
        
        # Agregamos contenedores para RAM y CPU
        resultados = {nombre: {"accuracy": [], "f1": [], "roc_auc": [], "tiempo": [], "ram": [], "cpu": []} for nombre in modelos}
        
        print(f"Procesando 5-Fold Cross Validation ({estrategia})...")
        
        fold_idx = 1
        for train_index, test_index in skf.split(X, y):
            X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
            y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]
            
            y_train_fold_enc, y_test_fold_enc, _ = codificar_target_seguro(y_train_fold, y_test_fold)
            
            X_train_enc, X_test_enc, _, _ = aplicar_encoding_seguro(X_train_fold, X_test_fold, columnas_categoricas)
            X_train_scaled, X_test_scaled, _ = aplicar_escalado_seguro(X_train_enc, X_test_enc, columnas_numericas)
            
            # Aplicación dinámica y segura de SMOTE
            if estrategia == "SMOTE":
                min_muestras = pd.Series(y_train_fold_enc).value_counts().min()
                if min_muestras > 1:
                    vecinos_k = min(5, min_muestras - 1)
                    smote = SMOTE(random_state=42, k_neighbors=vecinos_k)
                    X_train_final, y_train_final = smote.fit_resample(X_train_scaled, y_train_fold_enc)
                else:
                    X_train_final, y_train_final = X_train_scaled, y_train_fold_enc
            else:
                X_train_final, y_train_final = X_train_scaled, y_train_fold_enc
            
            for nombre, modelo in modelos.items():
                
                # Preparamos el monitor de CPU (lo ponemos en ceros para este bloque)
                proceso.cpu_percent(interval=None) 
                
                inicio = time.perf_counter()
                
                # Entrenamos y evaluamos
                modelo.fit(X_train_final, y_train_final)
                y_pred = modelo.predict(X_test_scaled)
                y_proba = modelo.predict_proba(X_test_scaled)
                
                tiempo_total = time.perf_counter() - inicio
                
                # Capturamos el consumo exacto al terminar el modelo
                cpu_usage = proceso.cpu_percent(interval=None)
                ram_usage = proceso.memory_info().rss / (1024 ** 3) # Convertimos a Gigabytes
                
                acc = accuracy_score(y_test_fold_enc, y_pred)
                f1 = f1_score(y_test_fold_enc, y_pred, average='macro')
                
                try:
                    num_clases = len(np.unique(y_train_final))
                    if num_clases == 2:
                        roc = roc_auc_score(y_test_fold_enc, y_proba[:, 1])
                    else:
                        roc = roc_auc_score(y_test_fold_enc, y_proba, multi_class='ovr', average='macro')
                except ValueError:
                    roc = np.nan 
                
                resultados[nombre]["accuracy"].append(acc)
                resultados[nombre]["f1"].append(f1)
                resultados[nombre]["roc_auc"].append(roc)
                resultados[nombre]["tiempo"].append(tiempo_total)
                resultados[nombre]["ram"].append(ram_usage)
                resultados[nombre]["cpu"].append(cpu_usage)

                # REPORTE Y GUARDADO POR ESTRATEGIA (Fold Individual)
                filas_csv.append({
                    "Dataset": nombre_dataset,
                    "Estrategia": estrategia,
                    "Modelo": nombre,
                    "Total_Folds": 5,
                    "Fold_Actual": f"Fold_{fold_idx}", 
                    "Accuracy": round(acc, 4),
                    "F1_Score": round(f1, 4),
                    "ROC_AUC": round(roc, 4) if not np.isnan(roc) else "N/A",
                    "Tiempo_s": round(tiempo_total, 4),
                    "RAM_GB": round(ram_usage, 4),
                    "CPU_Usage_%": round(cpu_usage, 2),
                    "Fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
            fold_idx += 1

        # REPORTE PROMEDIADO EN CONSOLA Y GUARDADO DE CSV
        print("\n" + "-"*85)
        print(f"RESULTADOS FINALES PROMEDIADOS ({estrategia})")
        print("-" * 85)
        print(f"{'Modelo':<15} | {'Accuracy':<8} | {'F1-Score':<8} | {'ROC-AUC':<8} | {'Tiempo':<8} | {'RAM (GB)':<8} | {'CPU (%)'}")
        print("-" * 85)
        
        for nombre in modelos:
            acc_mean = np.mean(resultados[nombre]['accuracy'])
            f1_mean = np.mean(resultados[nombre]['f1'])
            roc_mean = np.nanmean(resultados[nombre]['roc_auc'])
            tiempo_mean = np.mean(resultados[nombre]['tiempo'])
            ram_mean = np.mean(resultados[nombre]['ram'])
            cpu_mean = np.mean(resultados[nombre]['cpu'])
            
            print(f"{nombre:<15} | {acc_mean*100:>5.2f}%   | {f1_mean*100:>5.2f}%   | {roc_mean*100:>5.2f}%   | {tiempo_mean:.4f}s | {ram_mean:>6.4f}GB | {cpu_mean:>5.2f}%")
            
            # Guardamos también la fila del PROMEDIO FINAL
            filas_csv.append({
                "Dataset": nombre_dataset,
                "Estrategia": estrategia,
                "Modelo": nombre,
                "Total_Folds": 5,
                "Fold_Actual": "PROMEDIO",
                "Accuracy": round(acc_mean, 4),
                "F1_Score": round(f1_mean, 4),
                "ROC_AUC": round(roc_mean, 4) if not np.isnan(roc_mean) else "N/A",
                "Tiempo_s": round(tiempo_mean, 4),
                "RAM_GB": round(ram_mean, 4),
                "CPU_Usage_%": round(cpu_mean, 2),
                "Fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })

    # Guardado fuera de los bucles para consolidar todo
    df_resultados = pd.DataFrame(filas_csv)
    nombre_base = nombre_dataset.replace('.csv', '')
    ruta_salida = RESULTS_DIR / f"Baseline_{nombre_base}_COMPLETO.csv"
    
    df_resultados.to_csv(ruta_salida, index=False)
    print(f"\nResultados guardados exitosamente en: {ruta_salida}")

if __name__ == "__main__":
    preparar_directorios()
    nombre_archivo, ruta_dataset, target_col = menu_interactivo()
    
    try:
        df = cargar_dataset(ruta_dataset)
        df_limpio, registro_limpieza = limpiar_dataset(df, target_col)
        
        df_limpio.reset_index(drop=True, inplace=True)
        
        X = df_limpio.drop(columns=[target_col])
        y = df_limpio[target_col]
        
        columnas_categoricas = registro_limpieza["columnas_categoricas"]
        columnas_numericas = registro_limpieza["columnas_numericas"]
        
        evaluar_modelos_tabular(X, y, columnas_categoricas, columnas_numericas, nombre_archivo)
        
    except Exception as e:
        print(f"Error al ejecutar el baseline: {e}")