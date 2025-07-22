from sqlalchemy import create_engine
import urllib
from dotenv import load_dotenv
import os


# Cargar las variables de entorno del archivo .env
load_dotenv()

def get_mysql_engine():
    """
    Retorna un engine SQLAlchemy para conectarse a la base de datos MySQL usando variables del entorno.
    """
    username = os.getenv('DB_USER')
    password = os.getenv('DB_PASSWORD')
    host = os.getenv('DB_HOST')   
    database = os.getenv('DB_NAME')
    driver = 'ODBC Driver 17 for SQL Server'

    if not all([username, password, host,  database]):
        raise Exception("❌ Faltan variables en el archivo .env")

    connection_string = urllib.parse.quote_plus(
        f'DRIVER={{{driver}}};SERVER={host};DATABASE={database};UID={username};PWD={password}'
    )

# Crear motor de SQLAlchemy
    return create_engine(f'mssql+pyodbc:///?odbc_connect={connection_string}')

