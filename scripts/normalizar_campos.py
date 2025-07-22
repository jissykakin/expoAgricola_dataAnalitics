import pandas as pd
from sqlalchemy import create_engine
import re

from db_connection import get_mysql_engine

# Crear conexión
engine = get_mysql_engine()


# 2. Cargar datos
def load_data():
    # Base de datos de exportaciones
    query_export = "SELECT * FROM [AgroAnalyticsDB].[dbo].[exportaciones_agricolas]"
    df_export = pd.read_sql(query_export, engine)
    
    # Base de datos de producción
    query_prod = "SELECT DISTINCT [cultivo], [codigo_del_cultivo] FROM [AgroAnalyticsDB].[dbo].[produccion_agricola]"
    df_prod = pd.read_sql(query_prod, engine)
    
    return df_export, df_prod

# 3. Función para normalizar nombres
def normalize_name(name):
    if not isinstance(name, str):
        return ""
    # Eliminar espacios extras, caracteres especiales y convertir a minúsculas
    name = re.sub(r'[^a-zA-Z0-9áéíóúÁÉÍÓÚñÑ ]', '', str(name).strip().lower())
    # Eliminar prefijos comunes
    for prefix in ['el ', 'la ', 'los ', 'las ', 'un ', 'una ', 'unos ', 'unas ']:
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name

# 4. Procesamiento y unificación
def unify_datasets(df_export, df_prod):
    # Crear diccionario de cultivo a código
    prod_dict = {}
    for _, row in df_prod.iterrows():
        normalized = normalize_name(row['cultivo'])
        if normalized and row['codigo_del_cultivo']:
            if normalized not in prod_dict:  # Evitar duplicados
                prod_dict[normalized] = row['codigo_del_cultivo']
    
    # Buscar coincidencias en exportaciones
    df_export['codigo_del_cultivo'] = None
    
    for idx, row in df_export.iterrows():
        product_name = normalize_name(row['producto'])
        
        # Buscar coincidencia exacta
        if product_name in prod_dict:
            df_export.at[idx, 'codigo_del_cultivo'] = prod_dict[product_name]
            continue
        
        # Búsqueda aproximada para nombres similares
        for cultivo, codigo in prod_dict.items():
            if product_name in cultivo or cultivo in product_name:
                df_export.at[idx, 'codigo_del_cultivo'] = codigo
                break
    
    return df_export

# 5. Ejecución principal
def main():
    print("Cargando datos...")
    df_export, df_prod = load_data()
    
    print("Procesando y unificando datos...")
    df_unified = unify_datasets(df_export, df_prod)
    
    # Guardar resultados (opcional)
    output_file = "exportaciones_con_codigos.csv"
    df_unified.to_csv(output_file, index=False)
    print(f"Datos unificados guardados en {output_file}")
    
    # Opción para actualizar directamente en SQL Server
    update_db = input("¿Deseas actualizar la tabla en SQL Server? (s/n): ").lower()
    if update_db == 's':
        df_unified.to_sql('exportaciones_agricolas_enriquecidas', 
                         engine, 
                         if_exists='replace',
                         index=False)
        print("Tabla actualizada en SQL Server")

if __name__ == "__main__":
    main()