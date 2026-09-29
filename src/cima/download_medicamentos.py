# Leo csv
import argparse
import os
import sys
from io import BytesIO
import logging
from datetime import datetime
import requests
import pandas as pd
from cima_api import MyCimaAPI
from tqdm import tqdm
import json


def check_folder(path: str):
    if os.path.splitext(path)[1]:
        directory = os.path.dirname(path)
    else:
        directory = path
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def save_lista(name, lista, _type="a"):
    """
    Writes a list of elements to a file, separating them with colons.

    Input:
        name (str): Path to the file where data will be saved.
        lista (list): List of elements to be written to the file.
        _type (str): Write mode for the file ('a' for append, 'w' for write).

    Output:
        None: Writes to a file but does not return any value.
    """
    # Leer el contenido actual del archivo si el modo es append
    existing_elements = set()
    if _type == "a":
        try:
            with open(name, "r") as archivo:
                content = archivo.read()
                if content:
                    existing_elements = set(content.split(":"))
        except FileNotFoundError:
            # Si el archivo no existe, no hacer nada
            pass

    # Filtrar la lista para evitar elementos repetidos
    unique_elements = [
        elemento for elemento in lista if elemento not in existing_elements
    ]

    # Escribir los elementos únicos al archivo
    with open(name, _type) as archivo:
        for elemento in unique_elements:
            archivo.write(elemento + ":")


def save_json(text, filename):
    """
    Saves a given data structure in JSON format to a specified file.

    Input:
        text (any serializable data): Data to be saved in JSON format.
        filename (str): Path to the file where data will be saved.

    Output:
        None: Writes to a file but does not return any value.
    """
    with open(filename, "w") as file:
        json.dump(text, file, indent=4)


def download_medicamentos(
    df: pd.DataFrame,
    path_info: str,
    path_tecnico: str,
    path_fail: str,
    cima: MyCimaAPI,
    logger: logging.Logger,
):
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        nregistro = row.iloc[0]
        name = row.iloc[1]
        p_active = row.iloc[2]
        lista_mal_info = []
        lista_mal_tecnica = []
        logger.info(f"Descargando medicamento {name} con nº de registro {nregistro}")

        file_medicamento_info = os.path.join(path_info, f"{nregistro}_info.json")
        file_medicamento = os.path.join(path_tecnico, f"{nregistro}_tecnico.json")

        if os.path.isfile(file_medicamento_info) and os.path.isfile(file_medicamento):
            continue

        for index_try_info in range(5):
            text, status = cima.get_medicamento(cn="", nregister=nregistro)
            if (text == None or "error" in text) and index_try_info == 4:
                lista_mal_info.append(nregistro)
                logger.error(
                    f"Error peticion info {nregistro}, no se puede obtener informacion"
                )
                save_json({}, file_medicamento_info)
                break
            if status == 200 and not "error" in text:
                save_json(text, file_medicamento_info)
                logger.info(f"Info: {name}, nº: {nregistro} guardado")
                break
            else:
                logger.error(
                    f"Error peticion para el medicamento {nregistro}, con nombre {name}, try {index_try_info}/5"
                )

        for index_try_tecnico in range(5):
            text_completo, status = cima.get_info_medicamento_section(
                typeDoc=1, nregister=nregistro, section=""
            )
            if (
                text_completo == None or "error" in text_completo
            ) and index_try_tecnico == 4:
                lista_mal_tecnica.append(nregistro)
                logger.error(
                    f"Error peticion tecnica {nregistro}, no se puede obtener informacion"
                )
                save_json({}, file_medicamento)
                break
            if status == 200 and not "error" in text_completo:
                save_json(text_completo, file_medicamento)
                logger.info(f"Tecnica: {name}, nº: {nregistro} guardado")
                break
            else:
                logger.error(
                    f"Error peticion para el medicamento {nregistro}, con nombre {name}, try {index_try_tecnico}/5"
                )

    check_folder(path_fail)
    save_lista(os.path.join(path_fail, "lista_mal_info.txt"), lista_mal_info, "a")
    save_lista(os.path.join(path_fail, "lista_mal_tecnica.txt"), lista_mal_tecnica, "a")


def download_medicamentos_by_codigos(
    codigos: list,
    path_info: str,
    path_tecnico: str,
    path_fail: str,
    cima: MyCimaAPI,
    logger: logging.Logger,
):
    """Descarga la información y ficha técnica de un listado concreto de medicamentos por su código de registro.

    Input:
        codigos (list): Lista de códigos de registro (nregistro) a descargar.
        path_info (str): Ruta donde guardar la información del medicamento.
        path_tecnico (str): Ruta donde guardar la ficha técnica del medicamento.
        path_fail (str): Ruta donde guardar los medicamentos que han fallado.
        cima (MyCimaAPI): Instancia de la API de CIMA.
        logger (logging.Logger): Logger para registrar eventos.
    """
    lista_mal_info = []
    lista_mal_tecnica = []

    check_folder(path_info)
    check_folder(path_tecnico)

    for nregistro in tqdm(codigos, total=len(codigos)):
        nregistro = str(nregistro).strip()
        logger.info(f"Descargando medicamento con nº de registro {nregistro}")

        file_medicamento_info = os.path.join(path_info, f"{nregistro}_info.json")
        file_medicamento = os.path.join(path_tecnico, f"{nregistro}_tecnico.json")

        if os.path.isfile(file_medicamento_info) and os.path.isfile(file_medicamento):
            logger.info(f"Medicamento {nregistro} ya descargado, saltando")
            continue

        # Descargar info del medicamento
        for index_try_info in range(5):
            text, status = cima.get_medicamento(cn="", nregister=nregistro)
            if (text is None or "error" in text) and index_try_info == 4:
                lista_mal_info.append(nregistro)
                logger.error(
                    f"Error peticion info {nregistro}, no se puede obtener informacion"
                )
                save_json({}, file_medicamento_info)
                break
            if status == 200 and "error" not in text:
                save_json(text, file_medicamento_info)
                logger.info(f"Info: nº {nregistro} guardado")
                break
            else:
                logger.error(
                    f"Error peticion info para el medicamento {nregistro}, try {index_try_info}/5"
                )

        # Descargar ficha técnica
        for index_try_tecnico in range(5):
            text_completo, status = cima.get_info_medicamento_section(
                typeDoc=1, nregister=nregistro, section=""
            )
            if (
                text_completo is None or "error" in text_completo
            ) and index_try_tecnico == 4:
                lista_mal_tecnica.append(nregistro)
                logger.error(
                    f"Error peticion tecnica {nregistro}, no se puede obtener informacion"
                )
                save_json({}, file_medicamento)
                break
            if status == 200 and "error" not in text_completo:
                save_json(text_completo, file_medicamento)
                logger.info(f"Tecnica: nº {nregistro} guardado")
                break
            else:
                logger.error(
                    f"Error peticion tecnica para el medicamento {nregistro}, try {index_try_tecnico}/5"
                )

    check_folder(path_fail)
    save_lista(os.path.join(path_fail, "lista_mal_info.txt"), lista_mal_info, "a")
    save_lista(os.path.join(path_fail, "lista_mal_tecnica.txt"), lista_mal_tecnica, "a")


if __name__ == "__main__":

    cima = MyCimaAPI()

    parser = argparse.ArgumentParser(
        description="Script para descargar todos los medicamentos"
    )
    parser.add_argument(
        "--path_info",
        type=str,
        help="path donde guardar la info de los medicamentos",
        default="../../data/medicamentos/medicamentos_info/",
    )
    parser.add_argument(
        "--path_tecnico",
        type=str,
        help="path donde guardar la ficha tecnica de los medicamentos",
        default="../../data/medicamentos/medicamentos_tecnico/",
    )
    parser.add_argument(
        "--path_fail",
        type=str,
        help="path donde guardar los medicamentos que han fallado",
        default="../../data/medicamentos/",
    )
    parser.add_argument(
        "--codigos",
        type=str,
        nargs="+",
        help="Lista de códigos de registro (nregistro) a descargar. Ej: --codigos 06960 23622",
        default=None,
    )

    args = parser.parse_args()

    path_info = args.path_info
    path_tecnico = args.path_tecnico
    path_fail = args.path_fail
    codigos = args.codigos

    ############################################################################################################################################

    ##############
    #   Logger    #
    ##############

    logger = logging.getLogger("download_medicamentos")
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logger.info("Inicio del script")
    ############################################################################################################################################

    codigos = [ "06960" ]
    if codigos:
        # Descargar medicamentos concretos por código
        logger.info(f"Descargando medicamentos por códigos: {codigos}")
        download_medicamentos_by_codigos(
            codigos, path_info, path_tecnico, path_fail, cima, logger
        )
    else:
        # Descargar todos los medicamentos desde el listado completo
        url = "https://listadomedicamentos.aemps.gob.es/Medicamentos.xls"
        response = requests.get(url)

        # Asegurarse de que la solicitud fue exitosa
        if response.status_code == 200:
            # Leer los datos descargados en un DataFrame
            data = BytesIO(response.content)
            df = pd.read_excel(data)

            # Mostrar las primeras filas del DataFrame
            logger.info("DataFrame medicamentos descargado:")
        else:
            logger.error("Error al descargar el archivo:", response.status_code)

        check_folder(path_info)
        check_folder(path_tecnico)
        download_medicamentos(df, path_info, path_tecnico, path_fail, cima, logger)
