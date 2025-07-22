import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys, os

# Agregar ruta al módulo 'funciones'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'function')))

from getdate import get_date_from_period

from db_connection import get_mysql_engine  # ajusta si tu archivo está en otra carpeta

# --------------------
# CONEXIÓN A BASE DE DATOS
# --------------------
engine = get_mysql_engine()
query = "SELECT * FROM produccion_agricola"
df = pd.read_sql(query, con=engine)

print("✅ Datos cargados desde produccion_agricola:")
print(df.head())

print("✅ tipo de datos produccion_agricola:")
print(df.info())

# --------------------
# ELIMINAR DUPLICADOS
# --------------------
duplicados = df.duplicated()
if duplicados.sum() > 0:
    print(f"\n⚠️ Se encontraron {duplicados.sum()} filas duplicadas. Eliminando...")
    df = df.drop_duplicates()
    print(f"✅ Duplicados eliminados. Total actual: {len(df)}")
else:
    print("✅ No se encontraron duplicados.")

# --------------------
# VALORES NULOS
# --------------------
print("\n🔍 Valores nulos por columna:")
print(df.isnull().sum())

# --------------------
# CONVERSIÓN DE TIPOS NUMÉRICOS
# --------------------
numeric_cols = [
    'área_sembrada_(ha)', 'área_cosechada_(ha)',
    'producción_(t)', 'rendimiento_(t/ha)'
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# --------------------
# CREAR COLUMNA DE FECHA (si aplica)
# --------------------
# Aplicar función para construir fecha
df['fecha_estimada'] = df['periodo'].apply(get_date_from_period)
df['fecha_estimada'] = pd.to_datetime(df['fecha_estimada'], format='%Y-%m-%d', errors='coerce')

# Verificar resultado
print(df[['periodo', 'fecha_estimada']].drop_duplicates().head(10))
print("⏳ Filas con fecha estimada nula:", df['fecha_estimada'].isnull().sum())


# --------------------
# DETECCIÓN DE VALORES ATÍPICOS
# --------------------
plt.figure(figsize=(10, 4))
sns.boxplot(x=df['producción_(t)'])
plt.title("Outliers en producción_(t)")
plt.show()

plt.figure(figsize=(10, 4))
sns.boxplot(x=df['rendimiento_(t/ha)'])
plt.title("Outliers en Rendimiento")
plt.show()

# --------------------
# ESTADÍSTICAS BÁSICAS
# --------------------
print("\n📈 Estadísticas básicas:")
print(df[numeric_cols].describe())

# --------------------
# GUARDAR RESULTADO LIMPIO
# --------------------
df.to_csv('./outputs/produccion_agricola_limpia.csv', index=False)
print("\n✅ Datos limpios guardados en: outputs/produccion_agricola_limpia.csv")