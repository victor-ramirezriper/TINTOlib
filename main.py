import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold

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
    print("EXPERIMENTACIÓN TINTOLIB (K-FOLD - IMÁGENES ORIGINALES)")
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
    # CONFIGURACIÓN DE ESTRATEGIA ÚNICA (ORIGINAL)
    # ============================================================
    print(f"\nGenerando lote de imágenes bajo la estrategia: ORIGINAL (Pura, sin SMOTE visual).")

    N_FOLDS = 5
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    nombre_base = nombre_archivo.replace('.csv', '')
    nombre_lote_estrategia = f"{nombre_base}_Original"
    
    nombre_lote, ruta_lote_maestra = crear_carpeta_lote(nombre_lote_estrategia)
    print(f"\nCarpeta maestra asignada: {ruta_lote_maestra}")

    # BUCLE DE FOLDS
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        print("\n" + f" FOLD {fold} DE {N_FOLDS} ".center(60, "="))
        
        # 1. DIVISIÓN DEL PLIEGUE ACTUAL
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # 2. PREPROCESAMIENTO AISLADO (Cero Data Leakage)
        cols_categoricas = registro_limpieza["columnas_categoricas"]
        X_train_encoded, X_test_encoded, enc_aplicado, _ = aplicar_encoding_seguro(X_train, X_test, cols_categoricas)
        registro_limpieza["encoding_aplicado"] = enc_aplicado
        
        cols_numericas = registro_limpieza["columnas_numericas"]
        X_train_scaled, X_test_scaled, esc_aplicado = aplicar_escalado_seguro(X_train_encoded, X_test_encoded, cols_numericas)
        registro_limpieza["escalado_aplicado"] = esc_aplicado

        y_train_encoded, y_test_encoded, mapa_clases = codificar_target_seguro(y_train, y_test)
        registro_limpieza["mapa_clases"] = mapa_clases 

        # Asignación directa sin SMOTE visual
        X_train_final, y_train_final = X_train_scaled, y_train_encoded
        numero_features = len(X_train_final.columns)
        
        # 3. ANÁLISIS DEL FOLD ACTUAL
        df_train_temp = asegurar_target_al_final(X_train_final, y_train_final)
        df_test_temp = asegurar_target_al_final(X_test_scaled, y_test_encoded)
        
        info_train = analizar_dataset(df_train_temp, nombre_dataset=f"{nombre_lote_estrategia}_FOLD{fold}_TRAIN", columna_target=target_col)
        info_test = analizar_dataset(df_test_temp, nombre_dataset=f"{nombre_lote_estrategia}_FOLD{fold}_TEST", columna_target=target_col)
        
        # 4. GUARDAR CSVs VECTORIALES
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
            print(f"EJECUTANDO: {metodo.upper()} (FOLD {fold})")
            
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
                    df_features=X_train_final, 
                    series_target=y_train_final, 
                    nombre_lote=nombre_lote_fold, 
                    ruta_lote=ruta_imagenes_fold, 
                    informacion_dataset=info_train,
                    registro_preprocesamiento=registro_limpieza,
                    solo_transformar=False,
                    fold_actual=fold,
                    total_folds=N_FOLDS
                )
                
                # --- PRUEBA DEL FOLD ---
                generar_imagenes(
                    modelo=modelo,
                    metodo=nombre_metodo_test,
                    df_features=X_test_scaled, 
                    series_target=y_test_encoded, 
                    nombre_lote=nombre_lote_fold,
                    ruta_lote=ruta_imagenes_fold,
                    informacion_dataset=info_test,
                    registro_preprocesamiento=registro_limpieza,
                    solo_transformar=True,
                    fold_actual=fold,
                    total_folds=N_FOLDS
                )
                
            except Exception as e:
                print(f"\nError crítico al ejecutar {metodo.upper()} en Fold {fold}: {e}")
                print("Saltando al siguiente método...\n")

print("\n" + "=" * 60)
print(f"¡GENERACIÓN DE IMÁGENES FINALIZADA CON ÉXITO!")
print("=" * 60)