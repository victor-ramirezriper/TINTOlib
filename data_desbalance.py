
from pathlib import Path
import os
import pandas as pd
from src.project_config import DATA_DIR, RESULTS_DIR, preparar_directorios

RESUMEN = 'Resumen_Desbalance_Datasets.csv'
DETALLE = 'Detalle_Clases_Datasets.csv'


def elegir_opcion(maximo, mensaje):
    while True:
        opcion = input(mensaje).strip()
        if opcion.isdigit() and 0 <= int(opcion) <= maximo:
            return int(opcion)
        print(f'Introduce un número entre 0 y {maximo}.')


def analizar(ruta, target):
    df = pd.read_csv(ruta)
    if target not in df.columns:
        raise ValueError(f'No existe el target {target!r}')
    y = df[target]
    conteos = y.value_counts(dropna=True)
    if len(conteos) < 2:
        raise ValueError('El target necesita al menos dos clases distintas no nulas.')
    total = int(conteos.sum())
    mayor, menor = int(conteos.max()), int(conteos.min())
    resumen = {
        'Dataset': ruta.name, 'Target': target,
        'Filas_Originales': len(df), 'Filas_Target_Valido': total,
        'Filas_Target_Nulo': int(y.isna().sum()),
        'Num_Clases': len(conteos),
        'Clase_Mayoritaria': str(conteos.idxmax()), 'N_Mayoritaria': mayor,
        'Clase_Minoritaria': str(conteos.idxmin()), 'N_Minoritaria': menor,
        'Porcentaje_Minoritario': round(100 * menor / total, 4),
        'Ratio_Mayoritaria_Minoritaria': round(mayor / menor, 4),
    }
    detalle = [
        {'Dataset': ruta.name, 'Target': target, 'Clase': str(clase),
         'Observaciones': int(n), 'Porcentaje': round(100 * int(n) / total, 4)}
        for clase, n in conteos.items()
    ]
    return resumen, detalle


def actualizar_csv(ruta, nuevas_filas, dataset):
    """Reemplaza las filas del dataset, conservando todos los demás."""
    nuevos = pd.DataFrame(nuevas_filas)
    if ruta.exists():
        anterior = pd.read_csv(ruta, dtype={'Dataset': str}, encoding='utf-8-sig')
        if 'Dataset' not in anterior.columns:
            raise ValueError(f'El archivo {ruta} no tiene columna Dataset.')
        anterior = anterior.loc[anterior['Dataset'] != dataset]
        nuevos = pd.concat([anterior, nuevos], ignore_index=True)
    temporal = ruta.with_name(ruta.name + '.tmp')
    try:
        nuevos.to_csv(temporal, index=False, encoding='utf-8-sig')
        os.replace(temporal, ruta)
    finally:
        if temporal.exists():
            temporal.unlink()


def main():
    preparar_directorios()
    carpeta = Path(DATA_DIR)
    destino = Path(RESULTS_DIR)
    destino.mkdir(parents=True, exist_ok=True)

    while True:
        archivos = sorted(carpeta.glob('*.csv'), key=lambda p: p.name.lower())
        print('\n' + '=' * 65)
        print('ANÁLISIS DE DESBALANCE DE DATASETS — TINTOlib')
        print('=' * 65)
        if not archivos:
            print(f'No se encontraron archivos CSV en {carpeta}')
            return
        for numero, ruta in enumerate(archivos, start=1):
            print(f'  {numero}. {ruta.name}')
        print('  0. Salir')
        opcion = elegir_opcion(len(archivos), '\nSelecciona el dataset: ')
        if opcion == 0:
            print('Programa finalizado.')
            return

        ruta = archivos[opcion - 1]
        try:
            columnas = pd.read_csv(ruta, nrows=0).columns.tolist()
            if not columnas:
                raise ValueError('El CSV no contiene columnas.')
            print(f'\nColumnas de {ruta.name}:')
            for numero, columna in enumerate(columnas, start=1):
                print(f'  {numero}. {columna}')
            print('  0. Volver al menú')
            indice_target = elegir_opcion(len(columnas), '\nSelecciona la columna target: ')
            if indice_target == 0:
                continue
            target = columnas[indice_target - 1]
            resumen, detalle = analizar(ruta, target)
            print(f'\nDataset: {ruta.name}')
            print(f'Target: {target}')
            print(f'Clases: {resumen["Num_Clases"]}')
            print(f'Relación mayoritaria/minoritaria: {resumen["Ratio_Mayoritaria_Minoritaria"]:.2f}:1')
            for fila in detalle:
                print(f'  {fila["Clase"]}: {fila["Observaciones"]} ({fila["Porcentaje"]:.2f}%)')

            # Comprobar ambos destinos antes de actualizar cualquiera.
            for nombre in (RESUMEN, DETALLE):
                archivo = destino / nombre
                if archivo.exists():
                    cabecera = pd.read_csv(archivo, nrows=0).columns
                    if 'Dataset' not in cabecera:
                        raise ValueError(f'El archivo existente {nombre} no tiene columna Dataset.')
            actualizar_csv(destino / RESUMEN, [resumen], ruta.name)
            actualizar_csv(destino / DETALLE, detalle, ruta.name)
            print(f'\nGuardado en {destino.resolve()}')
            print('Se conservaron los resultados de otros datasets.')
        except (OSError, ValueError, pd.errors.ParserError, UnicodeError) as error:
            print(f'No se pudo analizar {ruta.name}: {error}')

        print('\n¿Deseas analizar otro dataset?')
        if elegir_opcion(1, '  1. Sí, volver al menú\n  0. Salir\nOpción: ') == 0:
            print('Programa finalizado.')
            return


if __name__ == '__main__':
    main()
