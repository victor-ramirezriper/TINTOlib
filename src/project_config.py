from pathlib import Path


def encontrar_raiz_proyecto():
    """
    Busca la raíz del proyecto a partir de la ubicación de este archivo.

    Se considera válida una carpeta que contenga:
    - src
    - Data
    - Image
    """
    carpeta_actual = Path(__file__).resolve().parent

    for carpeta in [carpeta_actual, *carpeta_actual.parents]:
        if (
            (carpeta / "src").is_dir()
            and (carpeta / "Data").is_dir()
            and (carpeta / "Image").is_dir()
        ):
            return carpeta

    raise FileNotFoundError(
        "No fue posible localizar la raíz del proyecto.\n"
        "Se esperaba encontrar las carpetas 'src', 'Data' e 'Image'."
    )


# ============================================================
# RUTAS PRINCIPALES DEL PROYECTO
# ============================================================

PROJECT_ROOT = encontrar_raiz_proyecto()

DATA_DIR = PROJECT_ROOT / "Data"
IMAGE_DIR = PROJECT_ROOT / "Image"
RESULTS_DIR = PROJECT_ROOT / "Results"
SRC_DIR = PROJECT_ROOT / "src"


def preparar_directorios():
    """
    Comprueba y crea los directorios que pueden faltar.
    """
    IMAGE_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)


def obtener_ruta_dataset(nombre_archivo):
    """
    Devuelve la ruta completa de un dataset ubicado en Data/.
    """
    ruta = DATA_DIR / nombre_archivo

    if not ruta.is_file():
        raise FileNotFoundError(
            f"No se encontró el dataset:\n{ruta}"
        )

    return ruta


def mostrar_configuracion():
    """
    Muestra las rutas detectadas del proyecto.
    """
    print("=" * 60)
    print("CONFIGURACIÓN DEL PROYECTO")
    print("=" * 60)
    print(f"Raíz del proyecto : {PROJECT_ROOT}")
    print(f"Datasets          : {DATA_DIR}")
    print(f"Imágenes          : {IMAGE_DIR}")
    print(f"Resultados        : {RESULTS_DIR}")
    print(f"Código fuente     : {SRC_DIR}")
    print("=" * 60)