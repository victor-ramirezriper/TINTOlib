import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Importar tus módulos existentes
from dataset_analyzer import cargar_dataset, analizar_dataset
from preprocessing import (
    limpiar_dataset, 
    aplicar_encoding_seguro, 
    aplicar_escalado_seguro, 
    codificar_target_seguro, 
    asegurar_target_al_final
)
from image_generator import crear_carpeta_lote, generar_imagenes
from model_config import crear_modelo, obtener_informacion_metodo

class LargeDatasetPipelineManager:
    def __init__(self, ruta_csv, metodo_tinto, test_size=0.2, random_state=42):
        """
        Gestor integral para ejecutar experimentos de transformación tabular-a-imagen 
        con TINTOlib en datasets grandes (10k+ instancias) utilizando Holdout (1 fold).
        """
        self.ruta_csv = ruta_csv
        self.metodo_nombre = metodo_tinto
        self.test_size = test_size
        self.random_state = random_state
        
        self.df_raw = None
        self.info_dataset = {}
        self.nombre_lote = None
        self.ruta_lote = None
        
        self.df_train_final = None
        self.df_test_final = None
        self.modelo_tinto = None

    def ejecutar_pipeline_completo(self):
        """Ejecuta de punta a punta el pipeline optimizado para datasets grandes."""
        
        print("=" * 60)
        print(f"INICIANDO PIPELINE DE GRAN ESCALA: {self.metodo_nombre.upper()}")
        print("=" * 60)

        # ------------------------------------------------------------------
        # PASO 1: Carga y Análisis Inicial del Dataset
        # ------------------------------------------------------------------
        print("\n[Paso 1/6] Cargando y analizando dataset original...")
        self.df_raw = cargar_dataset(self.ruta_csv)
        nombre_base = self.ruta_csv.split("/")[-1]
        self.info_dataset = analizar_dataset(self.df_raw, nombre_dataset=nombre_base)
        
        # Crear estructura de carpetas maestra por lote
        self.nombre_lote, self.ruta_lote = crear_carpeta_lote(nombre_base)
        print(f" > Lote asignado: {self.nombre_lote}")

        target_col = self.info_dataset["nombre_target"]

        # ------------------------------------------------------------------
        # PASO 2: Limpieza Dinámica (Nulos, Infinitos, Constantes)
        # ------------------------------------------------------------------
        print("\n[Paso 2/6] Limpiando dataset (nulos, infs, columnas constantes)...")
        df_limpio, registro_limpieza = limpiar_dataset(self.df_raw, target_col)

        # ------------------------------------------------------------------
        # PASO 3: División Train-Test (Holdout / 1 Fold Estratificado)
        # ------------------------------------------------------------------
        print("\n[Paso 3/6] Realizando división estratificada (Train / Test)...")
        X = df_limpio.drop(columns=[target_col])
        y = df_limpio[target_col]
        
        stratify_param = y if self.info_dataset.get("numero_clases", 0) > 1 else None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify_param
        )
        print(f" > Entrenamiento: {len(X_train)} muestras")
        print(f" > Prueba (Test): {len(X_test)} muestras")

        # ------------------------------------------------------------------
        # PASO 4: Preprocesamiento Seguro (Evitando Data Leakage)
        # ------------------------------------------------------------------
        print("\n[Paso 4/6] Aplicando transformaciones seguras (Encoding y Escalado)...")
        cat_cols = X_train.select_dtypes(include=["object", "category", "string"]).columns.tolist()
        num_cols = X_train.select_dtypes(include=["number"]).columns.tolist()

        # 4.1 One-Hot Encoding seguro
        X_train, X_test, encoding_aplicado, nombres_cats = aplicar_encoding_seguro(X_train, X_test, cat_cols)

        # 4.2 Escalado seguro (MinMaxScaler)
        X_train, X_test, escalado_aplicado = aplicar_escalado_seguro(X_train, X_test, num_cols)

        # 4.3 Codificación segura del Target (LabelEncoder para redes neuronales)
        y_train_encoded, y_test_encoded, mapa_clases = codificar_target_seguro(y_train, y_test)

        # Ensamblaje final de características con el target al final
        self.df_train_final = asegurar_target_al_final(X_train, y_train_encoded)
        self.df_test_final = asegurar_target_al_final(X_test, y_test_encoded)

        # Diccionario consolidado de preprocesamiento para tus logs
        registro_prep = {
            "columnas_numericas": num_cols,
            "columnas_categoricas": cat_cols,
            "encoding_aplicado": encoding_aplicado,
            "escalado_aplicado": escalado_aplicado
        }

        # ------------------------------------------------------------------
        # PASO 5: Instanciación Dinámica del Modelo TINTOlib
        # ------------------------------------------------------------------
        print(f"\n[Paso 5/6] Configurando modelo {self.metodo_nombre.upper()}...")
        # El número de características finales (excluyendo el target)
        numero_features_final = self.df_train_final.shape[1] - 1
        
        self.modelo_tinto = crear_modelo(self.metodo_nombre, numero_features_final)
        
        # Inyectar info de parámetros del método para los archivos de texto
        info_parametros = obtener_informacion_metodo(self.metodo_nombre, numero_features_final)
        self.info_dataset["parametros_metodo"] = info_parametros

        # ------------------------------------------------------------------
        # PASO 6: Generación Separada de Imágenes (Train y Test)
        # ------------------------------------------------------------------
        print(f"\n[Paso 6/6] Generando imágenes sintéticas...")
        
        # Separar features (X) y target (y) para el generador
        X_tr = self.df_train_final.drop(columns=[target_col])
        y_tr = self.df_train_final[target_col]

        X_te = self.df_test_final.drop(columns=[target_col])
        y_te = self.df_test_final[target_col]

        # 6.1 - Entrenamiento (Fit & Transform)
        print(" > Procesando imágenes de ENTRENAMIENTO...")
        generar_imagenes(
            modelo=self.modelo_tinto,
            metodo=f"{self.metodo_nombre.upper()}_TRAIN",
            df_features=X_tr,
            series_target=y_tr,
            nombre_lote=f"{self.nombre_lote}_TRAIN",
            ruta_lote=self.ruta_lote,
            informacion_dataset=self.info_dataset,
            registro_preprocesamiento=registro_prep,
            solo_transformar=False,
            fold_actual=1,
            total_folds=1
        )

        # 6.2 - Prueba / Test (Solo Transform - Ciego ante el fit)
        print(" > Procesando imágenes de PRUEBA (TEST)...")
        generar_imagenes(
            modelo=self.modelo_tinto,
            metodo=f"{self.metodo_nombre.upper()}_TEST",
            df_features=X_te,
            series_target=y_te,
            nombre_lote=f"{self.nombre_lote}_TEST",
            ruta_lote=self.ruta_lote,
            informacion_dataset=self.info_dataset,
            registro_preprocesamiento=registro_prep,
            solo_transformar=True, # <--- ¡Clave absoluta para evitar Data Leakage!
            fold_actual=1,
            total_folds=1
        )

        print("\n" + "=" * 60)
        print("¡PROCESO COMPLETADO EXITOSAMENTE PARA EL DATASET GRANDE!")
        print(f"Resultados guardados en: {self.ruta_lote}")
        print("=" * 60)