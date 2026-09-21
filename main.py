import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE  # <--- Para balancear antes de TINTOlib

from src.project_config import obtener_ruta_dataset, preparar_directorios, DATA_DIR
from src.dataset_analyzer import cargar_dataset, analizar_dataset
from src.preprocessing import (
    limpiar_dataset, 
    aplicar_encoding_seguro, 
    aplicar_escalado_seguro, 
    codificar_target_seguro,
    asegurar_target_al_final
)
from src.model_config import crear_modelo, obtener_informacion_metodo
from src.image_generator import generar_imagenes, crear_carpeta_lote

def menu_interactivo():
    print("=" * 60)
    print("EXPERIMENTACIÓN TINTOLIB (K-FOLD + AUTO-SMOTE)")
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
        
    opcion_tg = int(input("\nSelecciona el número de la variable objetivo (Target): ")) - 1
    target_col = columnas[opcion_tg]

    metodos = [
        "tinto", "igtd", "refined", "bargraph", "distancematrix", 
        "combination", "supertml", "featurewrap", "bie", "fotomics", "deepinsight"
    ]
    
    print("\nMétodos de transformación disponibles:")
    for i, met in enumerate(metodos, 1):
        print(f"  {i}. {met.upper()}")
    print("  12. TODOS LOS MÉTODOS (Ejecución en lote)")
        
    opcion_met = int(input("\nSelecciona el número del método a ejecutar: "))
    
    if opcion_met == 12:
        metodos_seleccionados = metodos
    else:
        metodos_seleccionados = [metodos[opcion_met - 1]]

    return nombre_archivo, ruta_dataset, target_col, metodos_seleccionados


if __name__ == "__main__":
    preparar_directorios()
    nombre_archivo, ruta_dataset, target_col, metodos_seleccionados = menu_interactivo()
    
    print("\n" + "=" * 60)
    print("INICIANDO LIMPIEZA BASE DEL DATASET")
    print("=" * 60)
    
    df = cargar_dataset(ruta_dataset)
    df_limpio, registro_limpieza = limpiar_dataset(df, target_col)
    df_limpio.reset_index(drop=True, inplace=True)
    
    X = df_limpio.drop(columns=[target_col])
    y = df_limpio[target_col]
    
    # ============================================================
    # DECISIÓN DE ESTRATEGIA (Original vs Original + SMOTE)
    # ============================================================
    conteo_clases = y.value_counts()
    clase_mayoritaria = conteo_clases.max()
    clase_minoritaria = conteo_clases.min()
    ratio = clase_mayoritaria / clase_minoritaria if clase_minoritaria > 0 else float('inf')
    
    estrategias = ["Original"]
    if ratio > 2.0:
        print(f"\nADVERTENCIA: Alto desbalanceo detectado (Ratio {ratio:.2f}).")
        print("MODO AUTOMÁTICO: Se generarán dos lotes de imágenes (ORIGINAL y SMOTE).")
        estrategias.append("SMOTE")
    else:
        print(f"\nDataset balanceado (Ratio {ratio:.2f}). Solo se generará el lote ORIGINAL.")

    N_FOLDS = 5
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    # ============================================================
    # BUCLE PRINCIPAL DE ESTRATEGIAS
    # ============================================================
    for estrategia in estrategias:
        print("\n" + "█" * 60)
        print(f"█ INICIANDO GENERACIÓN: ESTRATEGIA {estrategia.upper()}")
        print("█" * 60)

        # Modificamos el nombre del archivo virtualmente para que la carpeta refleje la estrategia
        nombre_base = nombre_archivo.replace('.csv', '')
        nombre_lote_estrategia = f"{nombre_base}_{estrategia}"
        
        nombre_lote, ruta_lote_maestra = crear_carpeta_lote(nombre_lote_estrategia)
        print(f"\nCarpeta maestra asignada: {ruta_lote_maestra}")

        # BUCLE DE FOLDS
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
            print("\n" + "" * 60)
            print(f" FOLD {fold} DE {N_FOLDS} | ESTRATEGIA: {estrategia.upper()}")
            print("" * 60)
            
            # 1. DIVISIÓN DEL PLIEGUE ACTUAL
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # 2. PREPROCESAMIENTO AISLADO
            cols_categoricas = registro_limpieza["columnas_categoricas"]
            X_train_encoded, X_test_encoded, enc_aplicado, _ = aplicar_encoding_seguro(X_train, X_test, cols_categoricas)
            registro_limpieza["encoding_aplicado"] = enc_aplicado
            
            cols_numericas = registro_limpieza["columnas_numericas"]
            X_train_scaled, X_test_scaled, esc_aplicado = aplicar_escalado_seguro(X_train_encoded, X_test_encoded, cols_numericas)
            registro_limpieza["escalado_aplicado"] = esc_aplicado

            y_train_encoded, y_test_encoded, mapa_clases = codificar_target_seguro(y_train, y_test)
            registro_limpieza["mapa_clases"] = mapa_clases 

            # 3. APLICACIÓN DE SMOTE (Solo si toca y SOLO en Train)
            if estrategia == "SMOTE":
                print("Aplicando SMOTE a los datos de entrenamiento...")
                smote = SMOTE(random_state=42)
                X_train_final, y_train_final = smote.fit_resample(X_train_scaled, y_train_encoded)
            else:
                X_train_final, y_train_final = X_train_scaled, y_train_encoded

            numero_features = len(X_train_final.columns)
            
            # 4. ANÁLISIS DE ESTE FOLD 
            df_train_temp = asegurar_target_al_final(X_train_final, y_train_final)
            df_test_temp = asegurar_target_al_final(X_test_scaled, y_test_encoded)
            
            info_train = analizar_dataset(df_train_temp, nombre_dataset=f"{nombre_lote_estrategia}_FOLD{fold}_TRAIN", columna_target=target_col)
            info_test = analizar_dataset(df_test_temp, nombre_dataset=f"{nombre_lote_estrategia}_FOLD{fold}_TEST", columna_target=target_col)
            
            # 5. GUARDAR CSVS VECTORIALES (Para garantizar equidad en la comparación posterior)
            ruta_csv_fold = ruta_lote_maestra / f"FOLD_{fold}" / "Datos_Tabulares"
            ruta_csv_fold.mkdir(parents=True, exist_ok=True)
            
            X_train_final.to_csv(ruta_csv_fold / "X_train_tabular.csv", index=False)
            X_test_scaled.to_csv(ruta_csv_fold / "X_test_tabular.csv", index=False)
            y_train_final.to_csv(ruta_csv_fold / "y_train_tabular.csv", index=False)
            y_test_encoded.to_csv(ruta_csv_fold / "y_test_tabular.csv", index=False)
            
            # ============================================================
            # GENERACIÓN DE IMÁGENES CON TINTOLIB
            # ============================================================
            ruta_imagenes_fold = ruta_lote_maestra / f"FOLD_{fold}" / "Imagenes"
            nombre_lote_fold = f"{nombre_lote}_FOLD_{fold}"
            
            for metodo in metodos_seleccionados:
                print("-" * 60)
                print(f"EJECUTANDO: {metodo.upper()} (FOLD {fold} | {estrategia.upper()})")
                
                try:
                    modelo = crear_modelo(metodo, numero_features)
                    info_metodo = obtener_informacion_metodo(metodo, numero_features)
                    info_train["parametros_metodo"] = info_metodo
                    info_test["parametros_metodo"] = info_metodo

                    nombre_metodo_train = f"EXP_{metodo.upper()}_TRAIN"
                    nombre_metodo_test = f"EXP_{metodo.upper()}_TEST"

                    # --- ENTRENAMIENTO DEL FOLD ---
                    generar_imagenes(
                        modelo=modelo,
                        metodo=nombre_metodo_train,
                        df_features=X_train_final,        # <--- TRAIN FINAL (Original o SMOTE)
                        series_target=y_train_final,      # <--- TARGET FINAL
                        nombre_lote=nombre_lote_fold,     
                        ruta_lote=ruta_imagenes_fold,     
                        informacion_dataset=info_train,
                        registro_preprocesamiento=registro_limpieza,
                        solo_transformar=False,
                        fold_actual=fold,                 # <--- NUEVO
                        total_folds=N_FOLDS               # <--- NUEVO
                    )
                    
                    # --- PRUEBA DEL FOLD ---
                    generar_imagenes(
                        modelo=modelo,
                        metodo=nombre_metodo_test,
                        df_features=X_test_scaled,        # <--- TEST SIEMPRE ORIGINAL
                        series_target=y_test_encoded,     # <--- TARGET TEST SIEMPRE ORIGINAL
                        nombre_lote=nombre_lote_fold,
                        ruta_lote=ruta_imagenes_fold,
                        informacion_dataset=info_test,
                        registro_preprocesamiento=registro_limpieza,
                        solo_transformar=True,
                        fold_actual=fold,                 # <--- NUEVO
                        total_folds=N_FOLDS               # <--- NUEVO
                    )
                    
                except Exception as e:
                    print(f"\nError crítico al ejecutar {metodo.upper()} en Fold {fold}: {e}")
                    print("Saltando al siguiente método...\n")

    print("\n" + "=" * 60)
    print(f"¡TODAS LAS ESTRATEGIAS FINALIZADAS CON ÉXITO!")
    print("=" * 60)