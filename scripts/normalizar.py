import pandas as pd
import unicodedata
from db_connection import get_mysql_engine

# --------------------------
# Función para normalizar texto
# --------------------------
def normalizar_texto(texto):
    if pd.isnull(texto):
        return ''
    texto = str(texto).lower().strip()
    texto = ''.join(
        c for c in unicodedata.normalize('NFD', texto)
        if unicodedata.category(c) != 'Mn'
    )  # Quitar tildes
    return texto

# --------------------------
# Conexión y lectura
# --------------------------
engine = get_mysql_engine()

df_exportaciones = pd.read_sql("SELECT * FROM exportaciones", con=engine)
df_departamentos = pd.read_sql("SELECT * FROM departamentos", con=engine)

# --------------------------
# Normalizar columnas de comparación
# --------------------------
df_exportaciones['dep_normalizado'] = df_exportaciones['departamento'].apply(normalizar_texto)
df_departamentos['dep_normalizado'] = df_departamentos['nombre_departamento'].apply(normalizar_texto)

# --------------------------
# Cruce por nombre normalizado
# --------------------------
df_merged = df_exportaciones.merge(
    df_departamentos[['codigo_dane_departamento', 'dep_normalizado']],
    on='dep_normalizado',
    how='left'
)

# --------------------------
# Verificar resultados
# --------------------------
print("Coincidencias encontradas:", df_merged['codigo_dane_departamento'].notnull().sum())
print("Departamentos no encontrados:")
print(df_merged[df_merged['codigo_dane_departamento'].isnull()]['departamento'].unique())

# --------------------------
# Guardar resultado limpio
# --------------------------
# df_merged.drop(columns=['dep_normalizado'], inplace=True)
# df_merged.to_sql('exportaciones_limpias', con=engine, if_exists='replace', index=False)
print("✅ Tabla 'exportaciones_limpias' cargada con códigos de departamento.")


df_no_match = df_merged[df_merged['codigo_dane_departamento'].isnull()]
print(df_no_match['departamento'].value_counts())