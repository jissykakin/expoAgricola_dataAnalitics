import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from db_connection import get_mysql_engine

# Crear conexión
engine = get_mysql_engine()

# Consultar tabla
query = "SELECT * FROM exportaciones_2"
df = pd.read_sql(query, con=engine)

print("✅ Datos cargados:")
print(df.head())


print(df.info())


# ---------------------
# LIMPIEZA DE DATOS
# ---------------------

# Revisar valores nulos
print("\n🔍 Valores nulos por columna:")
print(df.isnull().sum())

# Conversión de columnas numéricas
df['exportaciones_en_valor_(miles_usd_fob)'] = pd.to_numeric(df['exportaciones_en_valor_(usd_fob)'], errors='coerce')
df['exportaciones_en_volumen_(toneladas)'] = pd.to_numeric(df['exportaciones_en_volumen_(toneladas)'], errors='coerce')


# Mapeo de nombres de mes en español a número
meses_dict = {
    'enero': '01', 'febrero': '02', 'marzo': '03', 'abril': '04',
    'mayo': '05', 'junio': '06', 'julio': '07', 'agosto': '08',
    'septiembre': '09', 'setiembre': '09',  # ambos posibles
    'octubre': '10', 'noviembre': '11', 'diciembre': '12'
}

# Normalizar y mapear columna mes
df['mes_normalizado'] = df['mes'].str.lower().str.strip().map(meses_dict)

# Crear columna de fecha si existen 'Año' y 'Mes'
if 'año' in df.columns and 'mes_normalizado' in df.columns:
    df['fecha'] = pd.to_datetime(df['año'].astype(str) + '-' + df['mes_normalizado'].astype(str) + '-01', errors='coerce')

    print(df['fecha'])

# Eliminar filas completamente vacías
df.dropna(how='all', inplace=True)


##eliminar dupliciados
duplicados = df.duplicated()
num_duplicados = duplicados.sum()

if num_duplicados > 0:
    print(f"\n⚠️ Se encontraron {num_duplicados} filas duplicadas.")
    print("Mostrando las primeras duplicadas:")
    print(df[duplicados].head())

    # Opcional: eliminar duplicados
    df = df.drop_duplicates()
    print(f"✅ Duplicados eliminados. Nuevo total de filas: {len(df)}")
else:
    print("\n✅ No se encontraron filas duplicadas.")

# Visualización de valores atípicos
plt.figure(figsize=(10, 4))
sns.boxplot(x=df['exportaciones_en_valor_(miles_usd_fob)'])
plt.title("Outliers en valor de exportaciones")
plt.show()

plt.figure(figsize=(10, 4))
sns.boxplot(x=df['exportaciones_en_volumen_(toneladas)'])
plt.title("Outliers en volumen de exportaciones")
plt.show()

# Estadísticas generales
print("\n📈 Estadísticas básicas:")
print(df[['exportaciones_en_valor_(miles_usd_fob)', 'exportaciones_en_volumen_(toneladas)']].describe())

# Guardar DataFrame limpio
df.to_csv('./outputs/exportaciones_limpias.csv', index=False)
print("\n✅ Exportación limpia guardada en: outputs/exportaciones_limpias.csv")