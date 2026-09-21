import os
import time
import json
import random
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
# 0. CONFIGURACIÓN Y REPRODUCIBILIDAD
# ============================================================
def fijar_semillas(seed=42):
    """Fija la semilla en todas las librerías para garantizar reproducibilidad exacta."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ============================================================
# 1. DATASET PERSONALIZADO (CERO FUGA DE DATOS)
# ============================================================
class TintoDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = Path(folder_path)
        self.transform = transform
        
        ruta_etiquetas = self.folder_path / "etiquetas.csv"
        if not ruta_etiquetas.exists():
            raise FileNotFoundError(f"Falta etiquetas.csv en {self.folder_path}")
            
        df_etiquetas = pd.read_csv(ruta_etiquetas)
        self.labels = df_etiquetas.iloc[:, 0].values 
        
        extensiones = {".png", ".jpg", ".jpeg", ".bmp"}
        
        # CORRECCIÓN: Usamos rglob('*') para que busque dentro de las subcarpetas de clases (0, 1, etc.)
        archivos_img = [f for f in self.folder_path.rglob('*') if f.suffix.lower() in extensiones]
        
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
    carpeta_experimentos = Path(IMAGE_DIR) / "Experimentos"
    if not carpeta_experimentos.exists():
        print("No se encontró la carpeta de Experimentos.")
        exit()

    lotes = [d.name for d in carpeta_experimentos.iterdir() if d.is_dir()]
    print("="*60)
    print("SELECCIÓN DE DATOS Y RED NEURONAL")
    print("="*60)
    for i, lote in enumerate(lotes, 1):
        print(f"  {i}. {lote}")
    
    opcion_lote = int(input("\nSelecciona el Lote: ")) - 1
    lote_seleccionado = lotes[opcion_lote]
    ruta_lote = carpeta_experimentos / lote_seleccionado

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
    metodos_a_evaluar = metodos_generados if opcion_met == 99 else [metodos_generados[opcion_met - 1]]

    redes_disponibles = ["ResNet18 (CNN)", "EfficientNet-B0 (CNN)", "ViT-Tiny (Transformer)"]
    print("\nArquitecturas de Redes:")
    for i, red in enumerate(redes_disponibles, 1):
        print(f"  {i}. {red}")
    
    opcion_red = int(input("\nSelecciona la arquitectura a usar: ")) - 1
    red_seleccionada = redes_disponibles[opcion_red]

    return lote_seleccionado, metodos_a_evaluar, red_seleccionada, ruta_lote

def inicializar_modelo(nombre_red, num_clases, device):
    if "ResNet18" in nombre_red:
        modelo = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        modelo.fc = nn.Linear(modelo.fc.in_features, num_clases)
    elif "EfficientNet-B0" in nombre_red:
        modelo = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        modelo.classifier[1] = nn.Linear(modelo.classifier[1].in_features, num_clases)
    elif "ViT-Tiny" in nombre_red:
        modelo = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=num_clases)
    return modelo.to(device)

def entrenar_kfold_por_carpetas(ruta_lote_base, metodo, red_seleccionada, n_splits=5, epochs=10, batch_size=8, lr=0.001):
    transformaciones = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    device = torch.device("cpu")
    
    # Contenedores de registro exhaustivo
    registro_folds = []
    registro_historial = []

    print(f"\nIniciando Validación Cruzada ({n_splits} Folds) con {red_seleccionada} para {metodo}...")

    for fold in range(1, n_splits + 1):
        print(f"   Procesando Fold {fold}/{n_splits}...")
        
        ruta_train = ruta_lote_base / f"FOLD_{fold}" / "Imagenes" / f"EXP_{metodo.upper()}_TRAIN"
        ruta_test = ruta_lote_base / f"FOLD_{fold}" / "Imagenes" / f"EXP_{metodo.upper()}_TEST"

        if not ruta_train.exists() or not ruta_test.exists():
            print(f"   Fold {fold} no encontrado. Saltando...")
            continue

        dataset_train = TintoDataset(ruta_train, transform=transformaciones)
        dataset_test = TintoDataset(ruta_test, transform=transformaciones)

        num_clases = len(np.unique(dataset_train.labels))

        train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

        modelo = inicializar_modelo(red_seleccionada, num_clases, device)
        criterio = nn.CrossEntropyLoss()
        optimizador = optim.Adam(modelo.parameters(), lr=lr)

        inicio_train = time.perf_counter()

        # Entrenamiento por Épocas con registro de Loss
        for epoch in range(epochs):
            modelo.train()
            train_loss_acum = 0.0
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizador.zero_grad()
                outputs = modelo(inputs)
                loss = criterio(outputs, labels)
                loss.backward()
                optimizador.step()
                train_loss_acum += loss.item() * inputs.size(0)

            train_loss = train_loss_acum / len(dataset_train)

            # Calcular Validation Loss (sin entrenar)
            modelo.eval()
            val_loss_acum = 0.0
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = modelo(inputs)
                    loss = criterio(outputs, labels)
                    val_loss_acum += loss.item() * inputs.size(0)
            
            val_loss = val_loss_acum / len(dataset_test)

            # Registrar época
            registro_historial.append({
                "Metodo": metodo,
                "Fold": fold,
                "Epoch": epoch + 1,
                "Train_Loss": round(train_loss, 4),
                "Val_Loss": round(val_loss, 4)
            })

        tiempo_train_fold = time.perf_counter() - inicio_train

        # Evaluación Final y Métricas
        inicio_eval = time.perf_counter()
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

        tiempo_eval_fold = time.perf_counter() - inicio_eval

        y_true, y_pred, y_proba = np.array(y_true), np.array(y_pred), np.array(y_proba)

        acc = accuracy_score(y_true, y_pred) * 100
        f1 = f1_score(y_true, y_pred, average='macro') * 100
        
        try:
            if num_clases == 2:
                roc = roc_auc_score(y_true, y_proba[:, 1]) * 100
            else:
                roc = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro') * 100
        except ValueError:
            roc = np.nan

        registro_folds.append({
            "Metodo": metodo,
            "Fold": fold,
            "Accuracy": acc,
            "F1_Score": f1,
            "ROC_AUC": roc,
            "Tiempo_Entrenamiento_s": tiempo_train_fold,
            "Tiempo_Evaluacion_s": tiempo_eval_fold
        })

    if not registro_folds:
        raise ValueError("No se procesaron folds válidos.")

    # Calcular Estadísticas Finales (Medias y Desviaciones Estándar)
    df_folds = pd.DataFrame(registro_folds)
    
    resultados_agrupados = {
        "Accuracy_Mean": df_folds["Accuracy"].mean(),
        "Accuracy_Std": df_folds["Accuracy"].std(),
        "F1_Score_Mean": df_folds["F1_Score"].mean(),
        "F1_Score_Std": df_folds["F1_Score"].std(),
        "ROC_AUC_Mean": df_folds["ROC_AUC"].mean(),
        "ROC_AUC_Std": df_folds["ROC_AUC"].std(),
        "Tiempo_Train_Mean": df_folds["Tiempo_Entrenamiento_s"].mean(),
        "Tiempo_Eval_Mean": df_folds["Tiempo_Evaluacion_s"].mean()
    }
    
    return resultados_agrupados, registro_folds, registro_historial

def guardar_ecosistema_resultados(lote, red_seleccionada, metodos, df_resumen, lista_folds, lista_historial, parametros):
    """Crea la estructura de carpetas estandarizada y guarda los 4 archivos de registro."""
    nombre_dataset = lote.split("_LOTE_")[0]
    nombre_red_limpio = red_seleccionada.split(' ')[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Crear estructura: Results / Dataset / Arquitectura_Fecha
    carpeta_base = RESULTS_DIR / nombre_dataset / f"{nombre_red_limpio}_{timestamp}"
    carpeta_base.mkdir(parents=True, exist_ok=True)
    
    # 1. Guardar Resumen General
    pd.DataFrame(df_resumen).to_csv(carpeta_base / "resultados_resumen.csv", index=False)
    
    # 2. Guardar Datos por Fold
    pd.DataFrame(lista_folds).to_csv(carpeta_base / "resultados_folds.csv", index=False)
    
    # 3. Guardar Curvas de Aprendizaje (Historial Epochs)
    pd.DataFrame(lista_historial).to_csv(carpeta_base / "historial_entrenamiento.csv", index=False)
    
    # 4. Guardar Configuración (Metadata del Experimento)
    with open(carpeta_base / "experiment.json", "w") as json_file:
        json.dump(parametros, json_file, indent=4)
        
    print(f"\nEcosistema de resultados guardado exitosamente en: {carpeta_base}")


if __name__ == "__main__":
    fijar_semillas(42)  # Garantiza resultados idénticos en múltiples ejecuciones
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    lote, metodos, red_seleccionada, ruta_lote = seleccionar_experimento_y_red()
    
    # Parámetros globales del experimento a registrar
    PARAMETROS_EXP = {
        "dataset_lote": lote,
        "red_neuronal": red_seleccionada,
        "fecha_ejecucion": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_splits_kfold": 5,
        "epochs": 10,
        "batch_size": 8,
        "learning_rate": 0.001,
        "optimizer": "Adam",
        "loss_function": "CrossEntropyLoss",
        "seed": 42,
        "metodos_evaluados": metodos
    }

    print("\n" + "="*60)
    print(f"PROCESANDO LOTE: {lote}")
    print(f"ARQUITECTURA: {red_seleccionada}")
    print("="*60)

    resumen_total = []
    folds_totales = []
    historial_total = []

    for metodo in metodos:
        print(f"\n--- Evaluando método: {metodo} ---")
        try:
            # Entrenamiento y recolección
            res_agrupados, res_folds, res_historial = entrenar_kfold_por_carpetas(
                ruta_lote, metodo, red_seleccionada, 
                n_splits=PARAMETROS_EXP["n_splits_kfold"], 
                epochs=PARAMETROS_EXP["epochs"],
                batch_size=PARAMETROS_EXP["batch_size"],
                lr=PARAMETROS_EXP["learning_rate"]
            )
            
            # Acumuladores para guardado masivo
            folds_totales.extend(res_folds)
            historial_total.extend(res_historial)
            
            # Formatear la fila del resumen con Media ± STD
            fila_resumen = {
                "Metodo_TINTO": metodo,
                "Accuracy": f"{res_agrupados['Accuracy_Mean']:.2f} ± {res_agrupados['Accuracy_Std']:.2f}",
                "F1_Score": f"{res_agrupados['F1_Score_Mean']:.2f} ± {res_agrupados['F1_Score_Std']:.2f}",
                "ROC_AUC": f"{res_agrupados['ROC_AUC_Mean']:.2f} ± {res_agrupados['ROC_AUC_Std']:.2f}",
                "Tiempo_Train_s": round(res_agrupados['Tiempo_Train_Mean'], 2),
                "Tiempo_Eval_s": round(res_agrupados['Tiempo_Eval_Mean'], 2)
            }
            resumen_total.append(fila_resumen)
            
            print(f"Resultados {metodo}: Acc={fila_resumen['Accuracy']}% | F1={fila_resumen['F1_Score']}% | ROC={fila_resumen['ROC_AUC']}%")
            
        except Exception as e:
            print(f" Error crítico evaluando {metodo}: {e}")

    # Guardar Ecosistema Completo si hubo éxito
    if resumen_total:
        guardar_ecosistema_resultados(
            lote, red_seleccionada, metodos, 
            resumen_total, folds_totales, historial_total, PARAMETROS_EXP
        )
    
    print("\nPROCESO FINALIZADO.")