
import os
from pathlib import Path


def encontrar_raiz_proyecto():
    """
    Localiza la raíz del repositorio en Windows, Linux o Colab.

    También permite definir una ubicación personalizada mediante
    la variable de entorno TINTO_PROJECT_ROOT.
    """
    raiz_configurada = os.environ.get("TINTO_PROJECT_ROOT")

    if raiz_configurada:
        raiz = Path(raiz_configurada).expanduser().resolve()

        if (raiz / "src").is_dir() and (raiz / "main.py").is_file():
            return raiz

        raise FileNotFoundError(
            f"TINTO_PROJECT_ROOT no apunta a un proyecto válido: {raiz}"
        )

    archivo_actual = Path(__file__).resolve()

    for carpeta in archivo_actual.parents:
        if (
            (carpeta / "src").is_dir()
            and (carpeta / "main.py").is_file()
        ):
            return carpeta

    raise FileNotFoundError(
        "No se encontró la raíz de TINTOlib. "
        "Verifica que el repositorio contenga main.py y src/."
    )


PROJECT_ROOT = encontrar_raiz_proyecto()

DATA_DIR = PROJECT_ROOT / "Data"
IMAGE_DIR = PROJECT_ROOT / "Image"
RESULTS_DIR = PROJECT_ROOT / "Results"
ANALYSIS_DIR = PROJECT_ROOT / "Analysis"
SRC_DIR = PROJECT_ROOT / "src"


def preparar_directorios():
    """Crea las carpetas de trabajo que no existan."""
    for ruta in (
        DATA_DIR,
        IMAGE_DIR,
        RESULTS_DIR,
        ANALYSIS_DIR,
    ):
        ruta.mkdir(parents=True, exist_ok=True)


def obtener_ruta_dataset(nombre_archivo):
    """Obtiene y valida la ruta de un CSV de Data/."""
    ruta = Path(nombre_archivo).expanduser()

    # Acepta tanto un nombre de archivo como una ruta explícita.
    if not ruta.is_absolute():
        ruta = DATA_DIR / ruta

    ruta = ruta.resolve()

    if not ruta.is_file():
        raise FileNotFoundError(
            f"No se encontró el dataset:\n{ruta}\n"
            f"Carpeta de datos configurada: {DATA_DIR}"
        )

    if ruta.suffix.lower() != ".csv":
        raise ValueError(f"Se esperaba un archivo CSV: {ruta}")

    return ruta


def mostrar_configuracion():
    """Imprime las rutas detectadas."""
    print("=" * 60)
    print("CONFIGURACIÓN DEL PROYECTO")
    print("=" * 60)
    print(f"Raíz       : {PROJECT_ROOT}")
    print(f"Datasets   : {DATA_DIR}")
    print(f"Imágenes   : {IMAGE_DIR}")
    print(f"Resultados : {RESULTS_DIR}")
    print(f"Análisis   : {ANALYSIS_DIR}")
    print(f"Código     : {SRC_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    preparar_directorios()
    mostrar_configuracion()
