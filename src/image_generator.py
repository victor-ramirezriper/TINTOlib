import csv
import time
import pandas as pd  # Necesario para exportar las etiquetas
from datetime import datetime
from pathlib import Path
from PIL import Image
from src.project_config import IMAGE_DIR

# ============================================================
# GESTIÓN DE CARPETAS DE EXPERIMENTOS (SISTEMA POR LOTE)
# ============================================================

def crear_carpeta_lote(nombre_dataset, carpeta_base=IMAGE_DIR):
    """
    Crea una carpeta maestra única para todo el lote de experimentos de un dataset.
    Ejemplo: Image/Experimentos/dataset_LOTE_01/
    """
    carpeta_base = Path(carpeta_base)
    carpeta_experimentos = carpeta_base / "Experimentos"
    carpeta_experimentos.mkdir(parents=True, exist_ok=True)
    
    nombre_limpio = nombre_dataset.replace(".csv", "")
    
    numero = 1
    while True:
        nombre_lote = f"{nombre_limpio}_LOTE_{numero:02d}"
        ruta_lote = carpeta_experimentos / nombre_lote
        
        if not ruta_lote.exists():
            ruta_lote.mkdir(parents=True, exist_ok=True)
            return nombre_lote, ruta_lote
        numero += 1

# ============================================================
# ANÁLISIS DE IMÁGENES
# ============================================================

def contar_imagenes(carpeta):
    extensiones = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    cantidad = 0
    for archivo in Path(carpeta).rglob("*"):
        if archivo.is_file() and archivo.suffix.lower() in extensiones:
            cantidad += 1
    return cantidad

def obtener_resoluciones(carpeta):
    extensiones = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    resoluciones = set()
    for archivo in Path(carpeta).rglob("*"):
        if archivo.is_file() and archivo.suffix.lower() in extensiones:
            try:
                with Image.open(archivo) as imagen:
                    resoluciones.add(imagen.size)
            except Exception:
                continue
    return resoluciones

def obtener_archivos_generados(carpeta):
    archivos = []
    for archivo in Path(carpeta).rglob("*"):
        if archivo.is_file():
            archivos.append(str(archivo.relative_to(carpeta)))
    return sorted(archivos)

# ============================================================
# INFORMACIÓN DE CONFIGURACIÓN
# ============================================================

def obtener_parametros_modelo(modelo):
    try:
        return modelo.get_params()
    except AttributeError:
        return {}
    except Exception:
        return {}

def formatear_parametros(parametros):
    if not parametros:
        return (
            "Los parámetros no están disponibles mediante get_params().\n"
            "Consultar la configuración definida en model_config.py."
        )
    texto = ""
    for nombre, valor in parametros.items():
        texto += f"{nombre}: {valor}\n"
    return texto.rstrip()

# ============================================================
# VERIFICACIÓN DEL RESULTADO
# ============================================================

def verificar_resultado(carpeta_salida, imagenes_esperadas, df_features=None, series_target=None):
    """Audita classification.csv, imágenes y etiquetas contra las entradas del fold.

    La asociación imagen-fila se comprueba mediante la fila de classification.csv
    que contiene su ruta, no mediante el nombre de la subcarpeta (00/01 no es clase).
    """
    import numpy as np
    from pathlib import PureWindowsPath

    carpeta_salida = Path(carpeta_salida)
    extensiones = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    resultado = {
        "carpeta_salida": str(carpeta_salida),
        "carpeta_existe": carpeta_salida.is_dir(),
        "imagenes_generadas": 0, "imagenes_esperadas": imagenes_esperadas,
        "cantidad_correcta": False, "resoluciones": set(),
        "resolucion_consistente": False, "archivos_generados": [],
        "etiquetas_correctas": False, "classification_correcto": False,
        "correspondencia_verificada": False, "errores_verificacion": [],
        "valido": False,
    }
    errores = resultado["errores_verificacion"]
    if not carpeta_salida.is_dir():
        errores.append("No existe la carpeta de imágenes")
        return resultado

    imagenes = [p for p in carpeta_salida.rglob("*")
                if p.is_file() and p.suffix.lower() in extensiones]
    resultado["imagenes_generadas"] = len(imagenes)
    resultado["cantidad_correcta"] = len(imagenes) == imagenes_esperadas
    resultado["archivos_generados"] = obtener_archivos_generados(carpeta_salida)
    if not resultado["cantidad_correcta"]:
        errores.append(f"Cantidad incorrecta: {len(imagenes)} imágenes, esperadas {imagenes_esperadas}")

    dimensiones = set()
    for ruta in imagenes:
        try:
            with Image.open(ruta) as im:
                im.verify()
            with Image.open(ruta) as im:
                dimensiones.add(im.size)
        except Exception as exc:
            errores.append(f"Imagen ilegible: {ruta.relative_to(carpeta_salida)} ({exc})")
    resultado["resoluciones"] = dimensiones
    resultado["resolucion_consistente"] = len(dimensiones) == 1
    if not resultado["resolucion_consistente"]:
        errores.append("No hay una resolución única de imágenes válidas")

    ruta_etiquetas = carpeta_salida / "etiquetas.csv"
    if not ruta_etiquetas.is_file():
        errores.append("Falta etiquetas.csv")
    else:
        try:
            etiquetas = pd.read_csv(ruta_etiquetas)
            ok = len(etiquetas) == imagenes_esperadas and etiquetas.shape[1] == 1
            if ok and series_target is not None:
                esperado = pd.Series(series_target).reset_index(drop=True)
                guardado = etiquetas.iloc[:, 0].reset_index(drop=True)
                ok = guardado.equals(esperado) or (
                    guardado.astype(str).equals(esperado.astype(str))
                )
            resultado["etiquetas_correctas"] = bool(ok)
            if not ok:
                errores.append("etiquetas.csv no coincide con las etiquetas de entrada en cantidad u orden")
        except Exception as exc:
            errores.append(f"No se pudo leer etiquetas.csv: {exc}")

    ruta_class = carpeta_salida / "classification.csv"
    if not ruta_class.is_file():
        errores.append("Falta classification.csv")
    else:
        try:
            clas = pd.read_csv(ruta_class)
            if "images" not in clas.columns:
                errores.append("classification.csv no contiene la columna images")
            elif len(clas) != imagenes_esperadas:
                errores.append(f"classification.csv contiene {len(clas)} filas, esperadas {imagenes_esperadas}")
            else:
                rutas = []
                for valor in clas["images"]:
                    if pd.isna(valor):
                        errores.append("Ruta vacía en classification.csv")
                        continue
                    # TINTOlib escribe rutas Windows incluso en archivos compartidos.
                    partes = PureWindowsPath(str(valor)).parts
                    if not partes or any(p in ("..", "") for p in partes):
                        errores.append(f"Ruta insegura en classification.csv: {valor}")
                        continue
                    destino = carpeta_salida.joinpath(*partes)
                    if destino.suffix.lower() not in extensiones:
                        errores.append(f"Ruta no corresponde a una imagen: {valor}")
                    if not destino.is_file():
                        errores.append(f"Imagen referenciada inexistente: {valor}")
                    rutas.append(destino.resolve())
                if len(set(rutas)) != len(rutas):
                    errores.append("Hay imágenes repetidas en classification.csv")
                if set(rutas) != {p.resolve() for p in imagenes}:
                    errores.append("Las imágenes de classification.csv no coinciden con las del directorio")
                columnas = [c for c in clas.columns if c != "images"]
                if df_features is None:
                    errores.append("No se suministraron características para cotejar classification.csv")
                elif len(df_features) != len(clas):
                    errores.append("Filas de características distintas a classification.csv")
                elif len(columnas) != len(set(columnas)):
                    errores.append("classification.csv tiene columnas de características duplicadas")
                else:
                    # Algunos métodos TINTOlib renombran una característica como
                    # "values" en classification.csv. Sólo aceptamos ese alias
                    # cuando hay exactamente una columna faltante y una extra,
                    # y DESPUÉS comprobamos todos los valores fila por fila.
                    esperado = df_features.reset_index(drop=True)
                    faltantes = sorted(set(esperado.columns) - set(columnas))
                    adicionales = sorted(set(columnas) - set(esperado.columns))
                    mapa = {}
                    if faltantes or adicionales:
                        if len(faltantes) == 1 and adicionales == ["values"] and "values" not in esperado.columns:
                            mapa = {"values": faltantes[0]}
                            print(f"[AUDITORÍA] Alias de TINTOlib: values -> {faltantes[0]} (se cotejarán valores)")
                        else:
                            errores.append(
                                f"Columnas distintas en classification.csv. "
                                f"Faltantes: {faltantes}; adicionales: {adicionales}."
                            )
                    obtenido = clas[columnas].rename(columns=mapa)
                    if not errores and set(obtenido.columns) == set(esperado.columns):
                        obtenido = obtenido[list(esperado.columns)].reset_index(drop=True)
                    else:
                        obtenido = None
                    if obtenido is not None:
                        for c in esperado.columns:
                            a = pd.to_numeric(esperado[c], errors="coerce")
                            b = pd.to_numeric(obtenido[c], errors="coerce")
                            if a.notna().all() and b.notna().all():
                                iguales = np.allclose(a.to_numpy(dtype=float), b.to_numpy(dtype=float), rtol=1e-6, atol=1e-8)
                            else:
                                iguales = esperado[c].astype(str).equals(obtenido[c].astype(str))
                            if not iguales:
                                errores.append(f"Valores o orden de filas distintos en classification.csv: {c}")
                                break
                resultado["classification_correcto"] = not any(
                    "classification.csv" in e or "Imagen referenciada" in e or "imágenes repetidas" in e
                    or "Las imágenes de" in e or "Ruta " in e or "características" in e
                    for e in errores
                )
        except Exception as exc:
            errores.append(f"Error al leer o comparar classification.csv: {exc}")

    resultado["correspondencia_verificada"] = (
        resultado["classification_correcto"] and resultado["etiquetas_correctas"]
    )
    resultado["valido"] = (
        resultado["cantidad_correcta"] and resultado["resolucion_consistente"]
        and resultado["correspondencia_verificada"] and not errores
    )
    return resultado

# ============================================================
# REGISTROS: TXT (DETALLADO) Y CSV (GLOBAL)
# ============================================================

def guardar_resultado_txt(ruta_txt, experimento, metodo, estado, fecha_inicio, fecha_fin, tiempo_total, tiempo_transformacion, tiempo_verificacion, informacion_dataset, registro_preprocesamiento, parametros, resultado, error=None):
    lineas = []
    lineas.append("=" * 70 + "\nREGISTRO DE EXPERIMENTO\n" + "=" * 70 + "\n\n")
    lineas.append("IDENTIFICACIÓN\n" + "-" * 70 + "\n")
    lineas.append(f"Lote/Experimento: {experimento}\nMétodo: {metodo}\nEstado: {estado}\n")
    lineas.append(f"Fecha de inicio: {fecha_inicio}\nFecha de finalización: {fecha_fin}\n")
    lineas.append(f"Tiempo de transformación: {tiempo_transformacion:.6f} segundos\n")
    lineas.append(f"Tiempo de verificación: {tiempo_verificacion:.6f} segundos\n")
    lineas.append(f"Tiempo total del experimento: {tiempo_total:.6f} segundos\n\n")

    lineas.append("DATASET\n" + "-" * 70 + "\n")
    if informacion_dataset:
        for clave, valor in informacion_dataset.items():
            lineas.append(f"{clave}: {valor}\n")
    else:
        lineas.append("Información del dataset no disponible.\n")
    lineas.append("\n")

    lineas.append("PREPROCESAMIENTO\n" + "-" * 70 + "\n")
    if registro_preprocesamiento:
        for clave, valor in registro_preprocesamiento.items():
            lineas.append(f"{clave}: {valor}\n")
    else:
        lineas.append("Información de preprocesamiento no disponible.\n")
    lineas.append("\n")

    lineas.append("CONFIGURACIÓN DEL MÉTODO\n" + "-" * 70 + "\n")
    lineas.append(formatear_parametros(parametros) + "\n\n")

    lineas.append("RESULTADOS\n" + "-" * 70 + "\n")
    lineas.append(f"Carpeta de salida: {resultado.get('carpeta_salida', '')}\n")
    lineas.append(f"Imágenes esperadas: {resultado.get('imagenes_esperadas', 0)}\n")
    lineas.append(f"Imágenes generadas: {resultado.get('imagenes_generadas', 0)}\n")
    lineas.append(f"Cantidad correcta: {resultado.get('cantidad_correcta', False)}\n")

    resoluciones = resultado.get("resoluciones", set())
    if resoluciones:
        resoluciones_texto = ", ".join(f"{ancho}x{alto}" for ancho, alto in sorted(resoluciones))
    else:
        resoluciones_texto = "No determinadas"
    
    lineas.append(f"Resoluciones encontradas: {resoluciones_texto}\n")
    lineas.append(f"Resolución consistente: {resultado.get('resolucion_consistente', False)}\n\n")
    lineas.append(f"Etiquetas correctas: {resultado.get('etiquetas_correctas', False)}\n")
    lineas.append(f"Classification correcto: {resultado.get('classification_correcto', False)}\n")
    lineas.append(f"Correspondencia imagen-fila-etiqueta: {resultado.get('correspondencia_verificada', False)}\n")
    for detalle in resultado.get("errores_verificacion", []):
        lineas.append(f"Verificación: {detalle}\n")


    lineas.append("ARCHIVOS GENERADOS\n" + "-" * 70 + "\n")
    archivos_generados = resultado.get("archivos_generados", [])
    if archivos_generados:
        for nombre in archivos_generados:
            lineas.append(f"- {nombre}\n")
    else:
        lineas.append("No se detectaron archivos.\n")
    lineas.append("\n")

    lineas.append("ERROR\n" + "-" * 70 + "\n")
    if error:
        lineas.append(f"Tipo: {type(error).__name__}\nMensaje: {error}\n")
    else:
        lineas.append("Ninguno.\n")
    lineas.append("\n")

    lineas.append("ESTADO FINAL\n" + "-" * 70 + "\n")
    lineas.append(f"Experimento válido: {resultado.get('valido', False)}\n")
    lineas.append(f"Estado: {estado}\n\n")
    lineas.append("=" * 70 + "\n")

    texto_completo = "".join(lineas)
    with open(ruta_txt, "w", encoding="utf-8") as archivo:
        archivo.write(texto_completo)

def guardar_resultado_csv(ruta_csv, datos_fila):
    ruta_csv = Path(ruta_csv)
    archivo_existe = ruta_csv.exists()
    
    # NUEVO: Columnas total_folds y fold_actual agregadas
    columnas = [
        "dataset_nombre", "dataset_muestras", "dataset_caracteristicas", "dataset_clases", 
        "dataset_desbalanceado",
        "prep_categoricas", "prep_numericas", "prep_encoding", "prep_escalado",
        "trans_metodo", "trans_parametros", "trans_resolucion", "trans_tiempo_seg",
        "lote", "total_folds", "fold_actual", "estado", "valido", "error", "ruta_salida"
    ]
    
    ruta_csv.parent.mkdir(parents=True, exist_ok=True)
    
    with open(ruta_csv, mode="a", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=columnas)
        if not archivo_existe:
            escritor.writeheader()
        escritor.writerow(datos_fila)

# ============================================================
# GENERACIÓN PRINCIPAL
# ============================================================

def generar_imagenes(
    modelo,
    metodo,
    df_features,      
    series_target,    
    nombre_lote,
    ruta_lote,
    informacion_dataset=None,
    registro_preprocesamiento=None,
    solo_transformar=False,
    fold_actual=1,     # <--- NUEVO PARÁMETRO
    total_folds=1      # <--- NUEVO PARÁMETRO
):
    if series_target is None or len(df_features) != len(series_target):
        raise ValueError("Las características y etiquetas deben tener igual número de filas")
    if len(df_features) == 0:
        raise ValueError("El DataFrame procesado está vacío. Verifica las etapas de limpieza.")

    metodo = metodo.strip().upper()

    carpeta_salida = ruta_lote / metodo
    carpeta_salida.mkdir(parents=True, exist_ok=True)
    
    # ==========================================================
    # GUARDAR EL TARGET COMO ETIQUETAS.CSV (Para evitar Leakage)
    # ==========================================================
    if series_target is not None:
        ruta_etiquetas = carpeta_salida / "etiquetas.csv"
        series_target.to_csv(ruta_etiquetas, index=False, header=True)

    ruta_txt = carpeta_salida / "resultado.txt"
    numero_muestras = len(df_features)
    tiempo_total_inicio = time.perf_counter()
    fecha_inicio = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    estado = "FALLIDO"
    error = None
    tiempo_transformacion = 0.0
    tiempo_verificacion = 0.0
    
    if informacion_dataset and "parametros_metodo" in informacion_dataset:
        parametros = informacion_dataset["parametros_metodo"]
    else:
        parametros = obtener_parametros_modelo(modelo)

    try:
        print(f"\n{'-' * 60}")
        print(f"Método: {metodo} | {'(SOLO TRANSFORM)' if solo_transformar else '(FIT & TRANSFORM)'}")
        print(f"Salida: {carpeta_salida}")
        print(f"Muestras: {numero_muestras}")
        print("Generando imágenes...")

        inicio_transformacion = time.perf_counter()

        # TINTOlib procesa ÚNICAMENTE las características (X)
        if solo_transformar:
            modelo.transform(df_features, str(carpeta_salida))
        else:
            modelo.fit_transform(df_features, str(carpeta_salida))

        tiempo_transformacion = time.perf_counter() - inicio_transformacion

        inicio_verificacion = time.perf_counter()
        resultado = verificar_resultado(carpeta_salida, numero_muestras, df_features, series_target)
        tiempo_verificacion = time.perf_counter() - inicio_verificacion

        if resultado["valido"]:
            estado = "COMPLETADO"

    except Exception as e:
        if "inicio_transformacion" in locals():
            tiempo_transformacion = time.perf_counter() - inicio_transformacion
        error = e
        
        inicio_verificacion = time.perf_counter()
        resultado = verificar_resultado(carpeta_salida, numero_muestras, df_features, series_target)
        tiempo_verificacion = time.perf_counter() - inicio_verificacion
        estado = "FALLIDO"

    fecha_fin = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tiempo_total = time.perf_counter() - tiempo_total_inicio

    inicio_registro = time.perf_counter()
    guardar_resultado_txt(
        ruta_txt=ruta_txt,
        experimento=nombre_lote,
        metodo=metodo,
        estado=estado,
        fecha_inicio=fecha_inicio,
        fecha_fin=fecha_fin,
        tiempo_total=tiempo_total,
        tiempo_transformacion=tiempo_transformacion,
        tiempo_verificacion=tiempo_verificacion,
        informacion_dataset=informacion_dataset,
        registro_preprocesamiento=registro_preprocesamiento,
        parametros=parametros,
        resultado=resultado,
        error=error
    )
    tiempo_registro = time.perf_counter() - inicio_registro

    dataset_nombre = informacion_dataset.get('nombre_dataset', 'Desconocido') if informacion_dataset else 'Desconocido'
    dataset_caracteristicas = informacion_dataset.get('caracteristicas', 0) if informacion_dataset else 0
    dataset_clases = informacion_dataset.get('numero_clases', 'N/A') if informacion_dataset else 'N/A'
    dataset_desbalanceado = informacion_dataset.get('requiere_balanceo', 'N/A') if informacion_dataset else 'N/A'

    if registro_preprocesamiento:
        prep_categoricas = len(registro_preprocesamiento.get('columnas_categoricas', []))
        prep_numericas = len(registro_preprocesamiento.get('columnas_numericas', []))
        prep_encoding = registro_preprocesamiento.get('encoding_aplicado', False)
        prep_escalado = registro_preprocesamiento.get('escalado_aplicado', False)
    else:
        prep_categoricas = prep_numericas = 0
        prep_encoding = prep_escalado = False

    resoluciones = resultado.get("resoluciones", set())
    resolucion_str = "x".join(map(str, list(resoluciones)[0])) if resoluciones else "N/A"
    
    # NUEVO: Se agregaron los valores de fold_actual y total_folds al diccionario
    datos_csv = {
        "dataset_nombre": dataset_nombre,
        "dataset_muestras": numero_muestras,
        "dataset_caracteristicas": dataset_caracteristicas,
        "dataset_clases": dataset_clases,
        "dataset_desbalanceado": dataset_desbalanceado,
        "prep_categoricas": prep_categoricas,
        "prep_numericas": prep_numericas,
        "prep_encoding": prep_encoding,
        "prep_escalado": prep_escalado,
        "trans_metodo": metodo,
        "trans_parametros": str(parametros),
        "trans_resolucion": resolucion_str,
        "trans_tiempo_seg": round(tiempo_transformacion, 6),
        "lote": nombre_lote,
        "total_folds": total_folds,  # <--- NUEVO
        "fold_actual": fold_actual,  # <--- NUEVO
        "estado": estado,
        "valido": resultado["valido"],
        "error": str(error) if error else "Ninguno",
        "ruta_salida": str(carpeta_salida)
    }
    
    ruta_csv_global = ruta_lote.parent / "registro_global_experimentos.csv"
    guardar_resultado_csv(ruta_csv_global, datos_csv)

    print("-" * 60)
    if estado == "COMPLETADO":
        print(f"ÉXITO | Tiempo total: {tiempo_total:.4f}s")
    else:
        print(f"ERROR | {error}")
    print("-" * 60)

    return {
        "errores_verificacion": resultado["errores_verificacion"],
        "correspondencia_verificada": resultado["correspondencia_verificada"],
        "lote": nombre_lote,
        "metodo": metodo,
        "estado": estado,
        "carpeta_salida": str(carpeta_salida),
        "valido": resultado["valido"],
        "tiempo_transformacion": tiempo_transformacion,
        "tiempo_registro_txt": tiempo_registro
    }