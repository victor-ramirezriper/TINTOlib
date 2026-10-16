import os
import time
import json
import random
import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from datetime import datetime
import psutil 

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import timm

from src.project_config import IMAGE_DIR, RESULTS_DIR

warnings.filterwarnings('ignore')

# ============================================================
# 0. REPRODUCIBILIDAD
# ============================================================
def fijar_semillas(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ============================================================
# 1. DATASET OPTIMIZADO EN RAM
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
        archivos_img = [f for f in self.folder_path.rglob('*') if f.suffix.lower() in extensiones]
        
        def extract_number(filepath):
            numeros = ''.join(filter(str.isdigit, filepath.stem))
            return int(numeros) if numeros else 0
            
        self.image_files = sorted(archivos_img, key=extract_number)
        
        if len(self.image_files) != len(self.labels):
            raise ValueError(f"Inconsistencia en {self.folder_path}: {len(self.image_files)} imágenes vs {len(self.labels)} etiquetas.")

        # Carga previa en RAM para evitar cuello de botella
        self.cached_images = []
        from PIL import Image
        for img_path in self.image_files:
            image = Image.open(img_path).convert('RGB')
            self.cached_images.append(image)

    def __len__(self):
        return len(self.cached_images)

    def __getitem__(self, idx):
        image = self.cached_images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

# ============================================================
# 2. FÁBRICA DE MODELOS (CNNs y Vision Transformers)
# ============================================================
def inicializar_modelo(nombre_red, num_clases, device):
    nombre_red = nombre_red.lower()
    if "resnet18" in nombre_red:
        modelo = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        modelo.fc = nn.Linear(modelo.fc.in_features, num_clases)
    elif "efficientnet" in nombre_red:
        modelo = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        modelo.classifier[1] = nn.Linear(modelo.classifier[1].in_features, num_clases)
    elif "vit" in nombre_red or "transformer" in nombre_red:
        modelo = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=num_clases)
    else:
        raise ValueError(f"Arquitectura no reconocida: {nombre_red}")
    return modelo.to(device)

# ============================================================
# 3. GESTOR PRINCIPAL DE ENTRENAMIENTO UNIFICADO
# ============================================================
class DeepLearningPipelineManager:
    def __init__(self, red_seleccionada="ResNet18", epochs=10, batch_size=32, lr=0.001):
        self.red_seleccionada = red_seleccionada
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.transformaciones = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def entrenar_holdout_gran_escala(self, ruta_lote_base, metodo):
        """
        Entrenamiento optimizado para 1 solo Fold (Holdout 80/20) 
        para datasets grandes (10k+ instancias).
        """
        print(f"\n[*] Iniciando entrenamiento Holdout (1 Fold) para {metodo} con {self.red_seleccionada}...")
        
        ruta_train = ruta_lote_base / f"EXP_{metodo.upper()}_TRAIN"
        ruta_test = ruta_lote_base / f"EXP_{metodo.upper()}_TEST"

        if not ruta_train.exists() or not ruta_test.exists():
            ruta_train = ruta_lote_base / "Imagenes" / f"EXP_{metodo.upper()}_TRAIN"
            ruta_test = ruta_lote_base / "Imagenes" / f"EXP_{metodo.upper()}_TEST"

        if not ruta_train.exists():
            raise FileNotFoundError(f"No se encontró la ruta de entrenamiento para {metodo} en {ruta_lote_base}")

        dataset_train = TintoDataset(ruta_train, transform=self.transformaciones)
        dataset_test = TintoDataset(ruta_test, transform=self.transformaciones)

        num_clases = len(np.unique(dataset_train.labels))

        train_loader = DataLoader(dataset_train, batch_size=self.batch_size, shuffle=True, pin_memory=True)
        test_loader = DataLoader(dataset_test, batch_size=self.batch_size, shuffle=False, pin_memory=True)

        modelo = inicializar_modelo(self.red_seleccionada, num_clases, self.device)
        
        pesos_clase = compute_class_weight('balanced', classes=np.unique(dataset_train.labels), y=dataset_train.labels)
        pesos_tensor = torch.tensor(pesos_clase, dtype=torch.float).to(self.device)
        
        criterio = nn.CrossEntropyLoss(weight=pesos_tensor)
        optimizador = optim.Adam(modelo.parameters(), lr=self.lr)

        proceso = psutil.Process(os.getpid())
        inicio_train = time.perf_counter()
        
        max_ram = 0
        cpu_percents = []
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)

        for epoch in range(self.epochs):
            modelo.train()
            train_loss_acum = 0.0
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizador.zero_grad()
                outputs = modelo(inputs)
                loss = criterio(outputs, labels)
                loss.backward()
                optimizador.step()
                train_loss_acum += loss.item() * inputs.size(0)
                
                meminfo = proceso.memory_info()
                max_ram = max(max_ram, meminfo.rss / (1024 ** 3))
                cpu_percents.append(proceso.cpu_percent(interval=None))

        tiempo_train = time.perf_counter() - inicio_train
        avg_cpu = sum(cpu_percents) / len(cpu_percents) if cpu_percents else 0
        max_vram = torch.cuda.max_memory_allocated(self.device) / (1024 ** 3) if self.device.type == 'cuda' else 0

        modelo.eval()
        y_true, y_pred, y_proba = [], [], []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(self.device)
                outputs = modelo(inputs)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)

                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
                y_proba.extend(probs.cpu().numpy())

        y_true, y_pred, y_proba = np.array(y_true), np.array(y_pred), np.array(y_proba)

        acc = accuracy_score(y_true, y_pred) * 100
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0) * 100
        
        try:
            if num_clases == 2:
                roc = roc_auc_score(y_true, y_proba[:, 1]) * 100
            else:
                roc = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro') * 100
        except ValueError:
            roc = np.nan

        resultado_final = {
            "Metodo": metodo,
            "Accuracy": round(acc, 2),
            "F1_Score": round(f1, 2),
            "ROC_AUC": round(roc, 2) if not np.isnan(roc) else 0.0,
            "Tiempo_Train_s": round(tiempo_train, 2),
            "Max_RAM_GB": round(max_ram, 2),
            "Avg_CPU_Percent": round(avg_cpu, 2),
            "Max_VRAM_GB": round(max_vram, 2)
        }

        print(f"[+] Éxito {metodo} -> Acc: {acc:.2f}% | F1: {f1:.2f}% | Tiempo: {tiempo_train:.2f}s")
        return resultado_final