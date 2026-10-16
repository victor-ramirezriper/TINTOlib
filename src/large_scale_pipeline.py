import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Importas tus funciones existentes
from dataset_analyzer import cargar_dataset, analizar_dataset
from image_generator import crear_carpeta_lote, generar_imagenes

class LargeDatasetPipelineManager:
    def __init__(self, ruta_csv, test_size=0.2, random_state=42):
        """
        Gestor de experimentos para datasets grandes (10k+ instancias).
        Utiliza un enfoque de partición única (Train/Test) para evitar la 
        sobrecarga computacional de la validación cruzada (K-Fold).
        """
        self.ruta_csv = ruta_csv
        self.test_size = test_size
        self.random_state = random_state
        
        self.df = None
        self.info_dataset = {}
        self.nombre_lote = None
        self.ruta_lote = None
        
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        self.preprocessor = None

    def cargar_y_analizar(self):
        """Paso 1: Carga y análisis estructural usando tus módulos previos."""
        print("[*] Cargando y validando dataset...")
        self.df = cargar_dataset(self.ruta_csv)
        
        nombre_base = self.ruta_csv.split("/")[-1]
        self.info_dataset = analizar_dataset(self.df, nombre_dataset=nombre_base)
        
        # Crear la estructura de carpetas por lote
        self.nombre_lote, self.ruta_lote = crear_carpeta_lote(nombre_base)
        print(f"[+] Lote creado: {self.nombre_lote} en {self.ruta_lote}")
        
        return self.info_dataset

    def preparar_division_datos(self):
        """Paso 2: División estratificada Train-Test asegurando rigor científico."""
        target_col = self.info_dataset["nombre_target"]
        
        X = self.df.drop(columns=[target_col])
        y = self.df[target_col]
        
        # Estratificar si hay clases definidas
        stratify_param = y if self.info_dataset.get("numero_clases", 0) > 0 else None
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify_param
        )
        
        print(f"[+] División realizada exitosamente:")
        print(f"    - Entrenamiento: {len(self.X_train)} muestras")
        print(f"    - Prueba (Test): {len(self.X_test)} muestras")

    def preprocesar_caracteristicas(self):
        """Paso 3: Preprocesamiento adaptado para evitar Data Leakage."""
        num_cols = self.info_dataset["columnas_numericas"]
        cat_cols = self.info_dataset["columnas_categoricas"]
        
        # Excluir el target si por error se coló
        target_col = self.info_dataset["nombre_target"]
        num_cols = [c for c in num_cols if c != target_col]
        cat_cols = [c for c in cat_cols if c != target_col]

        transformers = []
        if num_cols:
            transformers.append(("num", StandardScaler(), num_cols))
        if cat_cols:
            transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols))

        self.preprocessor = ColumnTransformer(transformers=transformers, remainder="passthrough")

        # Ajustar SÓLO con Train y transformar ambos
        print("[*] Aplicando preprocesamiento (Fit en Train, Transform en Test)...")
        X_train_arr = self.preprocessor.fit_transform(self.X_train)
        X_test_arr = self.preprocessor.transform(self.X_test)

        # Recuperar nombres de columnas si es posible o mantener como DataFrame
        self.X_train_processed = pd.DataFrame(X_train_arr)
        self.X_test_processed = pd.DataFrame(X_test_arr)
        
        # Registro auxiliar para el generador de imágenes
        self.registro_prep = {
            "columnas_numericas": num_cols,
            "columnas_categoricas": cat_cols,
            "encoding_aplicado": bool(cat_cols),
            "escalado_aplicado": bool(num_cols)
        }

    def ejecutar_generacion_tinto(self, modelo_tinto, nombre_metodo="TINTO"):
        """Paso 4: Generación de imágenes separadas para Train y Test."""
        print(f"\n--- Procesando conjunto de ENTRENAMIENTO para {nombre_metodo} ---")
        
        # Crear subcarpeta específica para Train dentro del lote si se requiere, 
        # o pasar directamente respetando tu estructura de image_generator
        ruta_lote_train = self.ruta_lote / "train"
        ruta_lote_train.mkdir(parents=True, exist_ok=True)
        
        # Generar Train (Fit & Transform)
        generar_imagenes(
            modelo=modelo_tinto,
            metodo=f"{nombre_metodo}_TRAIN",
            df_features=self.X_train_processed,
            series_target=self.y_train,
            nombre_lote=f"{self.nombre_lote}_TRAIN",
            ruta_lote=self.ruta_lote,
            informacion_dataset=self.info_dataset,
            registro_preprocesamiento=self.registro_prep,
            solo_transformar=False,
            fold_actual=1,
            total_folds=1
        )

        print(f"\n--- Procesando conjunto de PRUEBA (TEST) para {nombre_metodo} ---")
        # Generar Test (Solo Transform, usando el modelo ya ajustado en Train)
        generar_imagenes(
            modelo=modelo_tinto,
            metodo=f"{nombre_metodo}_TEST",
            df_features=self.X_test_processed,
            series_target=self.y_test,
            nombre_lote=f"{self.nombre_lote}_TEST",
            ruta_lote=self.ruta_lote,
            informacion_dataset=self.info_dataset,
            registro_preprocesamiento=self.registro_prep,
            solo_transformar=True, # <--- ¡Clave para evitar fuga de datos!
            fold_actual=1,
            total_folds=1
        )
        print("[+] ¡Proceso de generación para dataset grande completado con éxito!")