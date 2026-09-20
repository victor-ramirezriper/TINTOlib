import os
import time
import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm
import timm

from src.project_config import IMAGE_DIR, RESULTS_DIR

warnings.filterwarnings('ignore')

# ============================================================
# 1. DATASET PERSONALIZADO (CERO FUGA DE DATOS)
# ============================================================
class TintoDataset(Dataset):
    """
    Lee las imágenes generadas por TINTOlib y busca sus etiquetas en 'etiquetas.csv'.
    Garantiza que el Target nunca formó parte de la transformación de la imagen.
    """
    def __init__(self, folder_path, transform=None):
        self.folder_path = Path(folder_path)
        self.transform = transform
        
        ruta_etiquetas = self.folder_path / "etiquetas.csv"
        if not ruta_etiquetas.exists():
            raise FileNotFoundError(f"Falta etiquetas.csv en {self.folder_path}")
            
        df_etiquetas = pd.read_csv(ruta_etiquetas)
        self.labels = df_etiquetas.iloc[:, 0].values 
        
        extensiones = {".png", ".jpg", ".jpeg", ".bmp"}
        archivos_img = [f for f in self.folder_path.iterdir() if f.suffix.lower() in extensiones]
        
        def extract_number(filepath):
            numeros = ''.join(filter(str.isdigit, filepath.stem))
            return int(numeros) if numeros else 0
            
        self.image_files = sorted(archivos_img, key=extract_number)
        
        if len(self.image_files) != len(self.labels):
            raise ValueError(f"Inconsistencia en {self.folder_path}: {len(self.image_files)} imágenes vs {len(self.labels)} etiquetas.")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        from PIL import Image
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)

def seleccionar_experimento_y_red():
    """Menú de selección de lote, métodos y arquitectura de red neuronal."""
    carpeta_experimentos = Path(IMAGE_DIR) / "Experimentos"
    if not carpeta_experimentos.exists():
        print("No se encontró la carpeta de Experimentos.")
        exit()

    lotes = [d.name for d in carpeta_experimentos.iterdir() if d.is_dir()]
    if not lotes:
        print("No hay lotes generados.")
        exit()

    print("="*60)
    print("SELECCIÓN DE DATOS Y RED NEURONAL")
    print("="*60)
    for i, lote in enumerate(lotes, 1):
        print(f"  {i}. {lote}")
    
    opcion_lote = int(input("\nSelecciona el Lote: ")) - 1
    lote_seleccionado = lotes[opcion_lote]
    ruta_lote = carpeta_experimentos / lote_seleccionado

    # Buscar métodos dentro del FOLD_1 para asegurar que la estructura por folds existe
    ruta_fold1 = ruta_lote / "FOLD_1" / "Imagenes"
    if not ruta_fold1.exists():
        print("Error: No se encontró la estructura de Folds dentro de este lote.")
        exit()

    metodos_generados = [d.name.replace("EXP_", "").replace("_TRAIN", "") 
                         for d in ruta_fold1.iterdir() if d.is_dir() and d.name.endswith("_TRAIN")]

    print("\nMétodos disponibles:")
    for i, met in enumerate(metodos_generados, 1):
        print(f"  {i}. {met}")
    print("  99. TODOS LOS MÉTODOS")

    opcion_met = int(input("\nSelecciona el método (99 para todos): "))
    if opcion_met == 99:
        metodos_a_evaluar = metodos_generados
    else:
        metodos_a_evaluar = [metodos_generados[opcion_met - 1]]

    redes_disponibles = ["ResNet18 (CNN)", "EfficientNet-B0 (CNN)", "ViT-Tiny (Transformer)"]
    print("\nArquitecturas de Redes:")
    for i, red in enumerate(redes_disponibles, 1):
        print(f"  {i}. {red}")
    
    opcion_red = int(input("\nSelecciona la arquitectura a usar: ")) - 1
    red_seleccionada = redes_disponibles[opcion_red]

    return lote_seleccionado, metodos_a_evaluar, red_seleccionada, ruta_lote

def inicializar_modelo(nombre_red, num_clases, device):
    """Descarga y configura el modelo pre-entrenado seleccionado."""
    if "ResNet18" in nombre_red:
        modelo = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        modelo.fc = nn.Linear(modelo.fc.in_features, num_clases)
    elif "EfficientNet-B0" in nombre_red:
        modelo = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        modelo.classifier[1] = nn.Linear(modelo.classifier[1].in_features, num_clases)
    elif "ViT-Tiny" in nombre_red:
        modelo = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=num_clases)
    else:
        raise ValueError("Arquitectura no reconocida")
    
    return modelo.to(device)

def entrenar_kfold_por_carpetas(ruta_lote_base, metodo, red_seleccionada, n_splits=5, epochs=10):
    """
    Recorre directamente los Folds generados por el pipeline principal (main.py).
    Entrena con TRAIN y evalúa estrictamente con TEST en cada fold correspondiente.
    """
    transformaciones = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    device = torch.device("cpu") # Cambiar a "cuda" si usas GPU compatible
    resultados_fold = {"accuracy": [], "f1": [], "roc_auc": [], "tiempo": []}

    print(f"\nIniciando Validación Cruzada ({n_splits} Folds) con {red_seleccionada} para el método {metodo}...")

    for fold in range(1, n_splits + 1):
        print(f"   Procesando Fold {fold}/{n_splits}...")
        
        ruta_train = ruta_lote_base / f"FOLD_{fold}" / "Imagenes" / f"EXP_{metodo.upper()}_TRAIN"
        ruta_test = ruta_lote_base / f"FOLD_{fold}" / "Imagenes" / f"EXP_{metodo.upper()}_TEST"

        if not ruta_train.exists() or not ruta_test.exists():
            print(f"   ⚠️ Advertencia: No se encontraron las carpetas para el Fold {fold}. Saltando...")
            continue

        # Cargamos datasets usando nuestras etiquetas seguras
        dataset_train = TintoDataset(ruta_train, transform=transformaciones)
        dataset_test = TintoDataset(ruta_test, transform=transformaciones)

        num_clases = len(np.unique(dataset_train.labels))

        train_loader = DataLoader(dataset_train, batch_size=8, shuffle=True)
        test_loader = DataLoader(dataset_test, batch_size=8, shuffle=False)

        modelo = inicializar_modelo(red_seleccionada, num_clases, device)
        criterio = nn.CrossEntropyLoss()
        optimizador = optim.Adam(modelo.parameters(), lr=0.001)

        inicio_tiempo = time.perf_counter()

        # Entrenamiento por Épocas
        for epoch in range(epochs):
            modelo.train()
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizador.zero_grad()
                outputs = modelo(inputs)
                loss = criterio(outputs, labels)
                loss.backward()
                optimizador.step()

        tiempo_total = time.perf_counter() - inicio_tiempo

        # Evaluación en el Test ciego de este Fold
        modelo.eval()
        y_true, y_pred, y_proba = [], [], []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                outputs = modelo(inputs)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)

                y_true.extend(labels.numpy())
                y_pred.extend(preds.numpy())
                y_proba.extend(probs.numpy())

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_proba = np.array(y_proba)

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
        
        try:
            if num_clases == 2:
                roc = roc_auc_score(y_true, y_proba[:, 1])
            else:
                roc = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
        except ValueError:
            roc = np.nan

        resultados_fold["accuracy"].append(acc)
        resultados_fold["f1"].append(f1)
        resultados_fold["roc_auc"].append(roc)
        resultados_fold["tiempo"].append(tiempo_total)

    if not resultados_fold["accuracy"]:
        raise ValueError("No se pudieron procesar folds válidos para este método.")

    # Promediar métricas de los folds
    return {
        "Accuracy": np.mean(resultados_fold["accuracy"]) * 100,
        "F1_Score": np.mean(resultados_fold["f1"]) * 100,
        "ROC_AUC": np.nanmean(resultados_fold["roc_auc"]) * 100,
        "Tiempo_Promedio_Fold": np.mean(resultados_fold["tiempo"])
    }

def guardar_resultados(lote, red_seleccionada, resultados_lista):
    """Guarda los resultados en una subcarpeta específica del dataset."""
    nombre_dataset = lote.split("_LOTE_")[0]
    
    carpeta_salida = RESULTS_DIR / nombre_dataset
    carpeta_salida.mkdir(parents=True, exist_ok=True)
    
    nombre_archivo = f"{red_seleccionada.split(' ')[0]}_Results.csv"
    ruta_archivo = carpeta_salida / nombre_archivo

    df_nuevo = pd.DataFrame(resultados_lista)

    if ruta_archivo.exists():
        df_existente = pd.read_csv(ruta_archivo)
        df_final = pd.concat([df_existente, df_nuevo], ignore_index=True)
    else:
        df_final = df_nuevo

    df_final.to_csv(ruta_archivo, index=False)
    print(f"\nResultados guardados en: {ruta_archivo}")


if __name__ == "__main__":
    preparar_directorios = lambda: [RESULTS_DIR.mkdir(parents=True, exist_ok=True)]
    preparar_directorios()
    
    lote, metodos, red_seleccionada, ruta_lote = seleccionar_experimento_y_red()
    
    resultados_totales = []
    fecha_actual = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("\n" + "="*60)
    print(f"PROCESANDO LOTE: {lote}")
    print(f"ARQUITECTURA: {red_seleccionada}")
    print("="*60)

    for metodo in metodos:
        print(f"\n--- Evaluando método: {metodo} ---")
        try:
            metricas = entrenar_kfold_por_carpetas(ruta_lote, metodo, red_seleccionada, n_splits=5, epochs=10)
            
            print(f"Resultados {metodo}: Acc={metricas['Accuracy']:.2f}% | F1={metricas['F1_Score']:.2f}% | ROC={metricas['ROC_AUC']:.2f}%")
            
            resultados_totales.append({
                "Dataset_Lote": lote,
                "Metodo_TINTO": metodo,
                "Modelo_Red": red_seleccionada.split(' ')[0],
                "Accuracy": round(metricas["Accuracy"], 2),
                "F1_Score": round(metricas["F1_Score"], 2),
                "ROC_AUC": round(metricas["ROC_AUC"], 2),
                "Tiempo_Prom_Fold_s": round(metricas["Tiempo_Promedio_Fold"], 4),
                "Fecha": fecha_actual
            })
            
        except Exception as e:
            print(f" Error evaluando {metodo}: {e}")

    if resultados_totales:
        guardar_resultados(lote, red_seleccionada, resultados_totales)
    
    print("\nPROCESO FINALIZADO.")