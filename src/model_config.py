import os
import math
import multiprocessing

from TINTOlib.tinto import TINTO
from TINTOlib.igtd import IGTD
from TINTOlib.refined import REFINED
from TINTOlib.barGraph import BarGraph
from TINTOlib.distanceMatrix import DistanceMatrix
from TINTOlib.combination import Combination
from TINTOlib.supertml import SuperTML
from TINTOlib.featureWrap import FeatureWrap
from TINTOlib.bie import BIE
from TINTOlib.fotomics import Fotomics
from TINTOlib.deepInsight import DeepInsight


SEED = 1
PROBLEM = "classification"


def calcular_dimension_cuadrada(numero_features):
    """
    Calcula la dimensión mínima de una imagen cuadrada
    capaz de contener todas las características.
    """
    if numero_features <= 0:
        raise ValueError(
            "El número de características debe ser mayor que cero."
        )

    return math.ceil(math.sqrt(numero_features))


def calcular_scale_igtd(numero_features):
    """
    Calcula automáticamente la escala cuadrada para IGTD.

    Ejemplos:
        64 features -> [8, 8]
        65 features -> [9, 9]
    """
    dimension = calcular_dimension_cuadrada(numero_features)

    return [dimension, dimension]


def obtener_numero_procesadores():
    """
    Obtiene el número de procesadores lógicos disponibles.
    """
    procesadores = os.cpu_count()

    if procesadores is None:
        procesadores = multiprocessing.cpu_count()

    return procesadores


def crear_modelo(metodo, numero_features):
    """
    Crea una instancia del método de TINTOlib seleccionado.

    Parámetros
    ----------
    metodo : str
        Nombre del método.

    numero_features : int
        Número de características después del preprocesamiento.

    Retorna
    -------
    modelo
        Instancia configurada del método.
    """
    metodo = metodo.strip().lower()

    if numero_features <= 0:
        raise ValueError(
            "El número de características debe ser mayor que cero."
        )

    # ---------------------------------------------------------
    # TINTO
    # ---------------------------------------------------------
    if metodo == "tinto":
        return TINTO(
            problem=PROBLEM,
            blur=True,
            random_seed=SEED
        )

    # ---------------------------------------------------------
    # IGTD
    # ---------------------------------------------------------
    elif metodo == "igtd":
        scale = calcular_scale_igtd(numero_features)
        return IGTD(
            problem=PROBLEM,
            scale=scale,
            random_seed=SEED
        )

    # ---------------------------------------------------------
    # REFINED
    # ---------------------------------------------------------
    elif metodo == "refined":
        return REFINED(
           problem=PROBLEM,
           hcIterations=1,
          # n_processors=obtener_numero_procesadores(), # Ajuste dinámico de CPU
           n_processors=8,
           random_seed=SEED,
           verbose=True
        )

    # ---------------------------------------------------------
    # BarGraph
    # ---------------------------------------------------------
    elif metodo == "bargraph":
        return BarGraph(
            problem=PROBLEM
        )

    # ---------------------------------------------------------
    # DistanceMatrix
    # ---------------------------------------------------------
    elif metodo == "distancematrix":
        return DistanceMatrix(
            problem=PROBLEM
        )

    # ---------------------------------------------------------
    # Combination
    # ---------------------------------------------------------
    elif metodo == "combination":
        return Combination(
            problem=PROBLEM
        )

    # ---------------------------------------------------------
    # SuperTML
    # ---------------------------------------------------------
    elif metodo == "supertml":
        return SuperTML(
            problem=PROBLEM,
            random_seed=SEED
        )

    # ---------------------------------------------------------
    # FeatureWrap
    # ---------------------------------------------------------
    elif metodo == "featurewrap":
        dimension = calcular_dimension_cuadrada(numero_features)
        return FeatureWrap(
            problem=PROBLEM,
            size=(dimension, dimension), # Ajuste dinámico de dimensiones
            bins=10,
            zoom=1
        )

    # ---------------------------------------------------------
    # BIE
    # ---------------------------------------------------------
    elif metodo == "bie":
        return BIE(
            problem=PROBLEM,
            precision=32,
            zoom=1
        )

    # ---------------------------------------------------------
    # Fotomics
    # ---------------------------------------------------------
    elif metodo == "fotomics":
        image_dim = calcular_dimension_cuadrada(numero_features)
        return Fotomics(
            image_dim=image_dim,
            problem=PROBLEM,
            random_seed=SEED
        )

    # ---------------------------------------------------------
    # DeepInsight
    # ---------------------------------------------------------
    elif metodo == "deepinsight":
        image_dim = calcular_dimension_cuadrada(numero_features)
        return DeepInsight(
            image_dim=image_dim,
            problem=PROBLEM,
            random_seed=SEED
        )

    else:
        raise ValueError(
            f"Método no reconocido: '{metodo}'."
        )


def obtener_informacion_metodo(metodo, numero_features):
    """
    Devuelve información útil sobre la configuración
    automática del método.
    """
    metodo = metodo.strip().lower()

    informacion = {
        "metodo": metodo,
        "problema": PROBLEM,
        "semilla": SEED,
        "numero_features": numero_features
    }

    if metodo == "igtd":
        informacion["scale"] = calcular_scale_igtd(numero_features)

    elif metodo in ["fotomics", "deepinsight"]:
        informacion["image_dim"] = calcular_dimension_cuadrada(numero_features)
        
    elif metodo == "featurewrap":
        dim = calcular_dimension_cuadrada(numero_features)
        informacion["size"] = (dim, dim)
        
    elif metodo == "refined":
        informacion["n_processors"] = obtener_numero_procesadores()

    return informacion