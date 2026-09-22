import os
import time
import numpy as np
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm

from src.project_config import IMAGE_DIR, RESULTS_DIR


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

warnings.filterwarnings("ignore")

BATCH_SIZE = 8
LEARNING_RATE = 0.001
DEFAULT_EPOCHS = 15
SEED = 1

torch.manual_seed(SEED)
np.random.seed(SEED)


# ============================================================
# UTILIDADES
# ============================================================

def obtener_nombre_dataset(ruta_train):
    """
    Obtiene el nombre del dataset a partir de la ruta de imágenes.
    """
    ruta_train = Path(ruta_train)
    posibles_archivos = list(ruta_train.rglob("*.csv"))

    if posibles_archivos:
        return posibles_archivos[0].stem

    nombre_lote = ruta_train.parent.name
    if nombre_lote.startswith("EXP_"):
        return nombre_lote

    return "dataset_desconocido"


def limpiar_nombre_archivo(nombre):
    """
    Limpia caracteres problemáticos para utilizar un nombre
    como archivo.
    """
    caracteres_invalidos = '<>:"/\\|?*'
    for caracter in caracteres_invalidos:
        nombre = nombre.replace(caracter, "_")
    return nombre.strip()


def crear_directorio_resultados(lote):
    """
    Crea la estructura: Results/Experimentos/EXP_XX/
    """
    ruta = Path(RESULTS_DIR) / "Experimentos" / lote
    ruta.mkdir(parents=True, exist_ok=True)
    return ruta


def guardar_registro_csv(ruta_csv, registro):
    """
    Agrega un registro al CSV.
    """
    import csv

    ruta_csv = Path(ruta_csv)
    ruta_csv.parent.mkdir(parents=True, exist_ok=True)

    existe = ruta_csv.exists()
    campos = list(registro.keys())

    with open(ruta_csv, "a", newline="", encoding="utf-8-sig") as archivo:
        escritor = csv.DictWriter(
            archivo,
            fieldnames=campos,
            extrasaction="ignore"
        )
        if not existe:
            escritor.writeheader()
        escritor.writerow(registro)


# ============================================================
# SELECCIÓN DEL EXPERIMENTO
# ============================================================

def seleccionar_experimento():
    """
    Menú para elegir lote, método TINTOlib y arquitectura CNN.
    """
    carpeta_experimentos = Path(IMAGE_DIR) / "Experimentos"

    if not carpeta_experimentos.exists():
        print("No se encontró la carpeta de Experimentos.")
        return None

    lotes = sorted(
        [
            d.name
            for d in carpeta_experimentos.iterdir()
            if d.is_dir()
        ]
    )

    if not lotes:
        print("No hay lotes generados.")
        return None

    print("\n" + "=" * 60)
    print("SELECCIÓN DE LOTE DE IMÁGENES")
    print("=" * 60)

    for i, lote in enumerate(lotes, 1):
        print(f"  {i}. {lote}")

    while True:
        try:
            opcion_lote = int(input("\nSelecciona el Lote: "))
            if 1 <= opcion_lote <= len(lotes):
                break
            print("Opción fuera de rango.")
        except ValueError:
            print("Introduce un número válido.")

    lote_seleccionado = lotes[opcion_lote - 1]
    ruta_lote = carpeta_experimentos / lote_seleccionado

    metodos_train = sorted(
        [
            d.name
            for d in ruta_lote.iterdir()
            if d.is_dir() and d.name.endswith("_TRAIN")
        ]
    )

    if not metodos_train:
        print(f"No se encontraron métodos TRAIN en {ruta_lote}.")
        return None

    metodos_base = [
        m.replace("EXP_", "").replace("_TRAIN", "")
        for m in metodos_train
    ]

    print("\nMétodos generados encontrados en este lote:")
    for i, metodo in enumerate(metodos_base, 1):
        print(f"  {i}. {metodo}")

    while True:
        try:
            opcion_met = int(input("\nSelecciona el método a evaluar con la CNN: "))
            if 1 <= opcion_met <= len(metodos_base):
                break
            print("Opción fuera de rango.")
        except ValueError:
            print("Introduce un número válido.")

    metodo_seleccionado = metodos_base[opcion_met - 1]

    ruta_train = ruta_lote / f"EXP_{metodo_seleccionado}_TRAIN"
    ruta_test = ruta_lote / f"EXP_{metodo_seleccionado}_TEST"

    print("\nArquitecturas disponibles:")
    print("  1. ResNet-18")
    print("  2. EfficientNet-B0")

    while True:
        try:
            opcion_arq = int(input("\nSelecciona la arquitectura: "))
            if opcion_arq in [1, 2]:
                break
            print("Selecciona 1 o 2.")
        except ValueError:
            print("Introduce un número válido.")

    arquitectura = "efficientnet_b0" if opcion_arq == 2 else "resnet18"

    return lote_seleccionado, metodo_seleccionado, arquitectura, ruta_train, ruta_test


# ============================================================
# ENTRENAMIENTO Y EVALUACIÓN
# ============================================================

def entrenar_y_evaluar_cnn(
    ruta_train,
    ruta_test,
    arquitectura,
    lote,
    metodo,
    epochs=DEFAULT_EPOCHS
):

    print("\n" + "=" * 60)
    print(f"INICIANDO TRANSFER LEARNING CON {arquitectura.upper()}")
    print("=" * 60)

    ruta_train = Path(ruta_train)
    ruta_test = Path(ruta_test)

    if not ruta_train.exists():
        raise FileNotFoundError(f"No existe TRAIN: {ruta_train}")

    if not ruta_test.exists():
        raise FileNotFoundError(f"No existe TEST: {ruta_test}")

    nombre_dataset = limpiar_nombre_archivo(
        obtener_nombre_dataset(ruta_train)
    )

    ruta_resultados = crear_directorio_resultados(lote)
    ruta_csv = ruta_resultados / f"{nombre_dataset}.csv"
    ruta_modelos = ruta_resultados / "modelos"
    ruta_modelos.mkdir(parents=True, exist_ok=True)

    nombre_modelo = f"{metodo}_{arquitectura}.pth"
    ruta_modelo = ruta_modelos / nombre_modelo

    transformaciones = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    tiempo_carga_inicio = time.perf_counter()

    dataset_train = datasets.ImageFolder(
        str(ruta_train),
        transform=transformaciones
    )

    dataset_test = datasets.ImageFolder(
        str(ruta_test),
        transform=transformaciones
    )

    tiempo_carga = time.perf_counter() - tiempo_carga_inicio

    if dataset_train.class_to_idx != dataset_test.class_to_idx:
        raise ValueError(
            "\nEl mapeo de clases entre TRAIN y TEST no coincide.\n\n"
            f"TRAIN: {dataset_train.class_to_idx}\n"
            f"TEST : {dataset_test.class_to_idx}"
        )

    clases = dataset_train.classes
    num_clases = len(clases)
    num_train = len(dataset_train)
    num_test = len(dataset_test)

    print(f"Imágenes Train: {num_train} | Imágenes Test: {num_test}")
    print(f"Clases detectadas: {num_clases} ({clases})")

    dataloader_train = DataLoader(
        dataset_train,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    dataloader_test = DataLoader(
        dataset_test,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    tiempo_modelo_inicio = time.perf_counter()

    if arquitectura == "efficientnet_b0":
        modelo = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT
        )
        num_ftrs = modelo.classifier[1].in_features
        modelo.classifier[1] = nn.Linear(num_ftrs, num_clases)
    else:
        modelo = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )
        num_ftrs = modelo.fc.in_features
        modelo.fc = nn.Linear(num_ftrs, num_clases)

    tiempo_carga_modelo = time.perf_counter() - tiempo_modelo_inicio

    # Configuración automática de dispositivo (GPU si está disponible)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo = modelo.to(device)

    print(f"Dispositivo utilizado: {device}")

    criterio = nn.CrossEntropyLoss()
    optimizador = optim.Adam(
        modelo.parameters(),
        lr=LEARNING_RATE
    )

    registro_base = {
        "tipo_registro": "",
        "dataset": nombre_dataset,
        "lote": lote,
        "metodo": metodo,
        "arquitectura": arquitectura,
        "device": str(device),
        "epochs_configuradas": epochs,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "optimizer": "Adam",
        "loss_function": "CrossEntropyLoss",
        "transform_resize": "224x224",
        "normalizacion": "ImageNet",
        "num_clases": num_clases,
        "clases": "|".join(map(str, clases)),
        "imagenes_train": num_train,
        "imagenes_test": num_test,
        "tiempo_carga_dataset_s": round(tiempo_carga, 4),
        "tiempo_carga_modelo_s": round(tiempo_carga_modelo, 4),
        "epoca": "",
        "loss_train": "",
        "accuracy_train": "",
        "accuracy_test": "",
        "f1_macro": "",
        "roc_auc": "",
        "tiempo_entrenamiento_s": "",
        "modelo_guardado": "",
        "estado": "EN_PROCESO",
        "mensaje": ""
    }

    tiempo_inicio = time.perf_counter()

    for epoch in range(epochs):
        modelo.train()
        loss_train = 0.0
        correctos_train = 0

        loop = tqdm(
            dataloader_train,
            desc=f"Época [{epoch + 1}/{epochs}]",
            leave=False
        )

        for inputs, labels in loop:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizador.zero_grad()
            outputs = modelo(inputs)
            loss = criterio(outputs, labels)
            loss.backward()
            optimizador.step()

            loss_train += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correctos_train += torch.sum(preds == labels.data)

        loss_epoch = loss_train / len(dataset_train)
        acc_epoch = (correctos_train.double() / len(dataset_train)).item()

        print(
            f"Época {epoch + 1}/{epochs} | "
            f"Loss: {loss_epoch:.4f} | "
            f"Acc Train: {acc_epoch:.4f}"
        )

        registro_epoca = registro_base.copy()
        registro_epoca.update({
            "tipo_registro": "EPOCH",
            "epoca": epoch + 1,
            "loss_train": round(loss_epoch, 6),
            "accuracy_train": round(acc_epoch * 100, 4),
            "estado": "OK"
        })

        guardar_registro_csv(ruta_csv, registro_epoca)

    tiempo_entrenamiento = time.perf_counter() - tiempo_inicio

    print("\nEvaluando en conjunto de Prueba (TEST Ciego)...")
    modelo.eval()

    y_true = []
    y_pred = []
    y_proba = []

    tiempo_evaluacion_inicio = time.perf_counter()

    with torch.no_grad():
        for inputs, labels in dataloader_test:
            inputs = inputs.to(device)
            outputs = modelo(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            # CORREGIDO: labels y preds se envían al CPU antes de convertir a numpy
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_proba.extend(probs.cpu().numpy())

    tiempo_evaluacion = time.perf_counter() - tiempo_evaluacion_inicio

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_proba = np.array(y_proba)

    acc = accuracy_score(y_true, y_pred) * 100
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0) * 100

    try:
        if num_clases == 2:
            roc = roc_auc_score(y_true, y_proba[:, 1]) * 100
        else:
            roc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro") * 100
    except ValueError:
        roc = np.nan

    torch.save(
        {
            "model_state_dict": modelo.state_dict(),
            "architecture": arquitectura,
            "classes": clases,
            "class_to_idx": dataset_train.class_to_idx,
            "num_classes": num_clases,
            "dataset": nombre_dataset,
            "method": metodo,
            "epochs": epochs,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE
        },
        ruta_modelo
    )

    registro_final = registro_base.copy()
    registro_final.update({
        "tipo_registro": "FINAL",
        "epoca": epochs,
        "accuracy_test": round(acc, 4),
        "f1_macro": round(f1, 4),
        "roc_auc": round(roc, 4) if not np.isnan(roc) else "",
        "tiempo_entrenamiento_s": round(tiempo_entrenamiento, 4),
        "modelo_guardado": str(ruta_modelo),
        "estado": "COMPLETADO"
    })

    guardar_registro_csv(ruta_csv, registro_final)

    print("\n" + "=" * 75)
    print(f"RESULTADOS FINALES: {arquitectura.upper()} SOBRE IMÁGENES TINTOLIB")
    print("=" * 75)
    print(f"{'Dataset':<25} | {nombre_dataset}")
    print(f"{'Método':<25} | {metodo}")
    print(f"{'Arquitectura':<25} | {arquitectura}")
    print("-" * 75)
    print(f"{'Accuracy TEST':<25} | {acc:.2f}%")
    print(f"{'F1-Score Macro':<25} | {f1:.2f}%")
    if np.isnan(roc):
        print(f"{'ROC-AUC':<25} | No disponible")
    else:
        print(f"{'ROC-AUC':<25} | {roc:.2f}%")
    print(f"{'Tiempo entrenamiento':<25} | {tiempo_entrenamiento:.2f}s")
    print(f"{'Tiempo evaluación':<25} | {tiempo_evaluacion:.2f}s")
    print(f"{'CSV resultados':<25} | {ruta_csv}")
    print(f"{'Modelo guardado':<25} | {ruta_modelo}")
    print("=" * 75)

    return acc, f1, roc, tiempo_entrenamiento


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

if __name__ == "__main__":
    try:
        seleccion = seleccionar_experimento()
        if seleccion is None:
            exit()

        lote, metodo, arquitectura, ruta_train, ruta_test = seleccion

        if not ruta_train.exists():
            print(f"No se encontró TRAIN para {metodo}:\n{ruta_train}")
            exit()

        if not ruta_test.exists():
            print(f"No se encontró TEST para {metodo}:\n{ruta_test}")
            exit()

        entrenar_y_evaluar_cnn(
            ruta_train=ruta_train,
            ruta_test=ruta_test,
            arquitectura=arquitectura,
            lote=lote,
            metodo=metodo,
            epochs=DEFAULT_EPOCHS
        )

    except KeyboardInterrupt:
        print("\n\nEntrenamiento cancelado por el usuario.")
    except Exception as e:
        print(f"\nError durante el entrenamiento PyTorch:\n{e}")