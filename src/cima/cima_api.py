import json
import logging
import os

import requests

cima_url = os.getenv("CIMAL_URL", "https://cima.aemps.es/cima/rest/")
timeout_value = int(os.getenv("TIMEOUT", 10))

logger = logging.getLogger(__name__)


class MyCimaAPI:
    """API para Cima.

    Atributos
    ----------

    Constructor
    ------------
        MyCimaAPI()

    Métodos
    -------
        create_url:
        get_medicamentos:
        get_medicamentos_all
        get_medicamento:
        get_info_medicamento_section:
        get_info_secciones
        buscarEnFichaTecnica:
        registroCambios
    """

    def __init__(self):
        pass

    def create_url(self, kind, list_var={}):
        """Dado el tipo de url, crea el link para realizar la peticion.

        Parametros
        ----------
            kind: String
                Tipo de petición a realizar en la API. Ex: Eutirox

        Retorna
        -------
            url: String
                Url para realizar la peticion, Ex: https://cima.aemps.es/cima/rest/medicamentos?&nombre=EUTIROX

        """

        url = cima_url + kind

        for index, (key, value) in enumerate(list_var.items()):
            if isinstance(value, int):
                url += "&" + key + "=" + str(value)
            elif isinstance(value, str):
                url += "&" + key + "=" + value
        # print(url)
        return url

    def get_medicamentos(self, name=None, page=1):
        """Devuelve informacion del medicamento proporcionado, puede ser mas de un tipo, dependiendo de los gramos, mililitros etc.

        Parametros
        ----------
            name: String
                Nombre del medicamento. Ex: Eutirox
            page: String
                Nombre del medicamento. Ex: Eutirox

        Retorna
        -------
            text: json
                Contiene el medicamento encontrado que coincide con el nombre proporcionado.

            resp_code: int
                Codigo de respuesta de nuestra consulta. Ex: 200
        """

        dic_var = {"nombre": name}

        url = self.create_url("medicamentos?", dic_var)

        params = {
            "pagina": page,  # Setting the current page number
            "tamanioPagina": 25,  # Setting the number of items per page
        }

        # Send query,
        try:
            response = requests.get(url, params=params, timeout=timeout_value)
            resp_code = response.status_code
            resp_message = response.text

            # If all OK, get info
            products = None
            if resp_code == 200:
                text = json.loads(resp_message)
            else:
                return None, resp_code

            return text, resp_code
        except Exception as e:
            logger.error(f"Error get_medicamentos: {e}")
            return None, None

    def get_medicamentos_all(self, name=None):
        """Devuelve informacion de medicamentos que contienen el nombre proporcionado, puede ser mas de un tipo, dependiendo de los gramos, mililitros etc.

        Parametros
        ----------
            nombre: String
                Nombre del medicamento. Ex: Eutirox

        Retorna
        -------
            text: list
                Contiene la lista de medicamentos encontrado que coincide con el nombre proporcionado.

        """

        dic_var = {"nombre": name}

        url = self.create_url("medicamentos?", dic_var)

        # Send query,
        try:
            response = requests.get(url, timeout=timeout_value)
            resp_message = response.json()
            total_pages = (
                resp_message["totalFilas"]
                if (resp_message["totalFilas"] < resp_message["tamanioPagina"])
                else round(resp_message["totalFilas"] / resp_message["tamanioPagina"])
            )

            list_objects = []
            # Looping through each page
            for page in range(1, total_pages + 1):
                # Parameters for the request
                params = {
                    "pagina": page,  # Setting the current page number
                    "tamanioPagina": 25,  # Setting the number of items per page
                }

                # Making the request
                try:
                    response = requests.get(url, params=params)

                    # Check if the request was successful
                    if response.status_code == 200:
                        data = response.json()
                        if not data["resultados"]:
                            break
                        else:
                            list_objects += data["resultados"]
                    else:
                        print(f"Failed to retrieve data for page {page}, name")
                except Exception as e:
                    logger.error(f"Error en la pagina {page}: {e}")
                    print(f"Error en la pagina {page}: ", e)

            return list_objects
        except Exception as e:
            logger.error(f"Error get_medicamentos_all: {e}")
            return None, None

    def get_medicamento(self, cn=None, nregister=None):
        """Devuelve informacion del medicamento proporcionado, que coincide con el numero de registro, tambien cuenta con los principios activos.

        Parametros
        ----------
            cn: int
                Codigo nacional. Ex: ?
            nregister: int
                Numero de registro. Ex: 64014

        Retorna
        -------
            text: json
                Contiene el medicamento encontrado que coincide con el numero de registro.

            resp_code: int
                Codigo de respuesta de nuestra consulta. Ex: 200
        """

        dic_var = {"cn": cn, "nregistro": nregister}

        url = self.create_url("medicamento?", dic_var)
        try:
            response = requests.get(url, timeout=timeout_value)
            resp_code = response.status_code

            # If all OK, get info
            if resp_code == 200:
                text = response.json()
            else:
                return None, resp_code

            return text, resp_code
        except Exception as e:
            logger.error(f"Error get_medicamento: {e}")
            return None, None

    def get_info_medicamento_section(self, typeDoc=1, nregister=None, section=""):
        """Devuelve una lista de secciones con su informacion para el tipo de documento asociado al medicamento por número de registro.

        Parametros
        ----------
            typeDoc: int
                Tipo de documento. Ex: 1/2
                    1: Ficha técnica
                    2: Prospecto
            nregister: int
                Numero de registro. Ex: 64014
            section: String
                Seccion a solicitar. Ex: 4.8 or None

        Retorna
        -------
            text: json
                Contiene el medicamento encontrado que coincide con el numero de registro.

            resp_code: int
                Codigo de respuesta de nuestra consulta. Ex: 200
        """

        dic_var = {"nregistro": nregister, "seccion": section}
        headers = {"Accept": "application/json"}

        url = self.create_url("docSegmentado/contenido/" + str(typeDoc) + "?", dic_var)

        # Send query,
        try:
            response = requests.get(url, timeout=timeout_value, headers=headers)
            resp_code = response.status_code

            # If all OK, get info
            if resp_code == 200:
                text = response.json()
            else:
                return None, resp_code

            return text, resp_code
        except Exception as e:
            logger.error(f"Error get_info_medicamento_section: {e}")
            return None, None

    def get_info_secciones(self, typeDoc=1, nregister=None):
        """Devuelve una lista con las secciones existentes para un tipo de documento y un medicamento.

        Parametros
        ----------
            tipoDoc: int
                Tipo de documento. Ex: 1/2
                    1: Ficha técnica
                    2: Prospecto
            nregistro: int
                Numero de registro. Ex: 64014

        Retorna
        -------
            text: json
                Contiene el medicamento encontrado que coincide con el numero de registro.

            resp_code: int
                Codigo de respuesta de nuestra consulta. Ex: 200
        """

        dic_var = {
            "nregistro": nregister,
        }

        url = self.create_url("docSegmentado/secciones/" + str(typeDoc) + "?", dic_var)

        # Send query,
        try:
            response = requests.get(url, timeout=timeout_value)
            resp_code = response.status_code

            # If all OK, get info
            if resp_code == 200:
                text = response.json()
            else:
                return None, resp_code

            return text, resp_code
        except Exception as e:
            logger.error(f"Error get_info_secciones: {e}")
            return None, None

    def buscarEnFichaTecnica(self, body):
        """Recibe en formato JSON como cuerpo de la petición una serie de texto a buscar en las secciones y devuelve una lista de medicamentos.

        Parametros
        ----------
            body: List
                Parametros para realizar la busqueda. Ex: # body = [{"seccion":"4.1","texto":"cáncer","contiene":1}]

        Retorna
        -------
            text: list
                Contiene la lista de medicamentos encontrado que coincide con el body proporcionado.

        """

        url = self.create_url("buscarEnFichaTecnica")
        headers = {"Content-Type": "application/json"}

        # Realizar la solicitud POST
        try:
            response = requests.post(
                url, data=json.dumps(body), headers=headers, timeout=timeout_value
            )
            resp_message = response.json()

            total_pages = (
                resp_message["totalFilas"]
                if (resp_message["totalFilas"] < resp_message["tamanioPagina"])
                else round(resp_message["totalFilas"] / resp_message["tamanioPagina"])
            )

            list_objects = []

            # Looping through each page
            for page in range(1, total_pages + 1):
                # Parameters for the request
                params = {
                    "pagina": page,  # Setting the current page number
                    "tamanioPagina": 25,  # Setting the number of items per page
                }

                # Making the request
                try:
                    response = requests.post(
                        url, data=json.dumps(body), headers=headers, params=params
                    )

                    # Check if the request was successful
                    if response.status_code == 200:
                        data = response.json()
                        if not data["resultados"]:
                            break
                        else:
                            list_objects += data["resultados"]
                    else:
                        print(f"Failed to retrieve data for page {page}")
                except Exception as e:
                    logger.error(f"Error en la pagina {page}: {e}")

            return list_objects
        except Exception as e:
            logger.error(f"Error buscarEnFichaTecnica: {e}")
            return None, None

    def registroCambios(self, date, nregister=""):
        """Recibe en formato JSON como cuerpo de la petición una serie de texto a buscar en las secciones y devuelve una lista de medicamentos.

        Parametros
        ----------
            date: String
               Fecha desde la que se quiere conoecer los cambios. Ex: 11/11/2023
            nregistro: int
                Numero de registro. Ex: 64014

        Retorna
        -------
            text: list
                Contiene la lista de medicamentos que han recibido algun cambio desde la fecha indicada.
        """

        dic_var = {"fecha": date, "nregistro": nregister}

        url = self.create_url("registroCambios?", dic_var)

        try:
            # Send query,
            response = requests.post(url, timeout=timeout_value)
            resp_message = response.json()

            total_pages = (
                resp_message["totalFilas"]
                if (resp_message["totalFilas"] < resp_message["tamanioPagina"])
                else round(resp_message["totalFilas"] / resp_message["tamanioPagina"])
            )

            list_objects = []

            # Looping through each page
            for page in range(1, total_pages + 1):
                # Parameters for the request
                params = {
                    "pagina": page,  # Setting the current page number
                    "tamanioPagina": 25,  # Setting the number of items per page
                }

                # Making the request
                try:
                    response = requests.post(url, params=params)

                    # Check if the request was successful
                    if response.status_code == 200:
                        data = response.json()
                        if not data["resultados"]:
                            break
                        else:
                            list_objects += data["resultados"]
                    else:
                        print(f"Failed to retrieve data for page {page}")
                except Exception as e:
                    logger.error(f"Error en la pagina {page}: {e}")

            return list_objects
        except Exception as e:
            logger.error(f"Error registroCambios: {e}")
            return None, None
