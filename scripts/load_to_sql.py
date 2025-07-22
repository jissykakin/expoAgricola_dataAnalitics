import pandas as pd
from sqlalchemy import create_engine
import urllib

# Configuración de conexión
server = 'localhost'  # o IP/hostname de tu SQL Server
database = 'AgroAnalyticsDB'  # nombre de tu base de datos
username = 'sa'  # si usas autenticación SQL Server
password = 'b4st4rd0'
driver = 'ODBC Driver 17 for SQL Server'

# Codificar parámetros para SQLAlchemy
connection_string = urllib.parse.quote_plus(
    f'DRIVER={{{driver}}};SERVER={server};DATABASE={database};UID={username};PWD={password}'
)

# Crear motor de SQLAlchemy
engine = create_engine(f'mssql+pyodbc:///?odbc_connect={connection_string}')


# CARGAR CSV DE EXPORTACIONES
df_exportaciones = pd.read_csv('./data/Exportaciones_20250625.csv')

# Limpiar nombres de columnas (opcional)
df_exportaciones.columns = df_exportaciones.columns.str.strip().str.lower().str.replace(" ", "_")
df_exportaciones.to_sql('exportaciones_2', con=engine, if_exists='replace', index=False)
print("Exportaciones cargadas con éxito.")

# CARGAR EXCEL DE PRODUCCIÓN AGRÍCOLA
df_agricola = pd.read_excel('./data/Base agrícola 2019 - 2023.xlsx')

df_agricola.columns = df_agricola.columns.str.strip().str.lower().str.replace(" ", "_")


# -------------------------------
# CREAR TABLA PAISES
# -------------------------------
# CARGAR CSV DE EXPORTACIONES
df_country = pd.read_csv('./data/paises.csv')
print(df_country)
df_country.columns = df_country.columns.str.strip().str.lower().str.replace(" ", "_")
df_country.to_sql('country', con=engine, if_exists='replace', index=False)
print("✅ Tabla 'Paises' creada.")



# -------------------------------
# CREAR TABLA DEPARTAMENTOS / MUNICIPIOS
# -------------------------------

df_DANE = pd.read_csv('./data/Departamentos_y_Municipios.csv')
df_DANE.columns = df_DANE.columns.str.strip().str.lower().str.replace(" ", "_")
print(df_DANE)
df_departamentos = df_DANE[['region','código_dane_del_departamento', 'departamento']].drop_duplicates()
df_departamentos.columns = ['region','codigo_dane_departamento', 'departamento']
df_departamentos.to_sql('departamentos', con=engine, if_exists='replace', index=False)
print("✅ Tabla 'departamentos' creada.")

# -------------------------------
# CREAR TABLA MUNICIPIOS
# -------------------------------
df_municipios = df_DANE[['código_dane_del_municipio', 'municipio', 'código_dane_del_departamento']].drop_duplicates()
df_municipios.columns = ['codigo_dane_municipio', 'municipio', 'codigo_dane_departamento']
df_municipios['codigo_dane_municipio'] = df_municipios['codigo_dane_municipio'].astype(str).str.replace(',', '').str.replace('.', '')

df_municipios.to_sql('municipios', con=engine, if_exists='replace', index=False)
print("✅ Tabla 'municipios' creada.")

# Eliminar columnas de texto redundantes
df_agricola = df_agricola.drop(columns=['codigo_dane_departamento','departamento', 'municipio'])

# Renombrar columnas para consistencia
df_agricola.columns = df_agricola.columns.str.replace('á', 'a').str.replace('é', 'e') \
                                         .str.replace('í', 'i').str.replace('ó', 'o').str.replace('ú', 'u') \
                                         .str.replace('ñ', 'n')

# Guardar la tabla principal
df_agricola.to_sql('produccion', con=engine, if_exists='replace', index=False)
print("✅ Tabla 'produccion' cargada correctamente.")



