import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy import text 
import re
from unidecode import unidecode
import hashlib

from db_connection import get_mysql_engine

# Crear conexión
engine = get_mysql_engine()

# 2. Cargar datos
def load_data():
    # Base de datos de exportaciones
    query_export = "SELECT DISTINCT [producto] FROM [AgroAnalyticsDB].[dbo].[exportaciones_agricolas]"
    df_export = pd.read_sql(query_export, engine)
    
    # Base de datos de producción
    query_prod = "SELECT DISTINCT [cultivo] FROM [AgroAnalyticsDB].[dbo].[produccion_agricola]"
    df_prod = pd.read_sql(query_prod, engine)
    
    return df_export, df_prod

# 3. Función para normalizar nombres mejorada
def normalize_name(name):
    if not isinstance(name, str):
        return ""
    
    # Convertir a minúsculas y quitar acentos
    name = unidecode(str(name).strip().lower())
    
    # Eliminar caracteres especiales y múltiples espacios
    name = re.sub(r'[^a-zA-Z0-9 ]', '', name)
    name = re.sub(r'\s+', ' ', name)
    
    # Eliminar palabras comunes
    stop_words = {'de', 'del', 'la', 'las', 'los', 'el', 'y', 'en', 'con'}
    name = ' '.join([word for word in name.split() if word not in stop_words])
    
    return name

# 4. Crear catálogo unificado
def create_unified_catalog(df_export, df_prod):
    # Unificar todos los nombres de productos
    all_products = set(df_export['producto'].dropna().unique())
    all_products.update(df_prod['cultivo'].dropna().unique())
    
    # Crear DataFrame para el catálogo
    catalog = pd.DataFrame({'nombre_original': list(all_products)})
    
    # Normalizar nombres
    catalog['nombre_normalizado'] = catalog['nombre_original'].apply(normalize_name)
    
    # Generar código único basado en hash del nombre normalizado
    catalog['codigo_producto'] = catalog['nombre_normalizado'].apply(
        lambda x: hashlib.md5(x.encode()).hexdigest()[:8].upper()
    )
    
    # Ordenar por nombre normalizado
    catalog = catalog.sort_values('nombre_normalizado').reset_index(drop=True)
    
    # Asignar código numérico secuencial
    catalog['id_producto'] = catalog.index + 1
    
    return catalog[['id_producto', 'codigo_producto', 'nombre_original', 'nombre_normalizado']]

# 5. Actualizar tablas originales
def update_original_tables(catalog, engine):
    # Actualizar tabla de exportaciones
    update_export = text("""
    UPDATE e
    SET e.id_producto = c.id_producto,
        e.codigo_producto = c.codigo_producto
    FROM [AgroAnalyticsDB].[dbo].[exportaciones_agricolas] e
    JOIN [AgroAnalyticsDB].[dbo].[catalogo_productos] c
    ON e.producto = c.nombre_original
    """)
    
    # Actualizar tabla de producción
    update_prod = text("""
    UPDATE p
    SET p.id_producto = c.id_producto,
        p.codigo_producto = c.codigo_producto
    FROM [AgroAnalyticsDB].[dbo].[produccion_agricola] p
    JOIN [AgroAnalyticsDB].[dbo].[catalogo_productos] c
    ON p.cultivo = c.nombre_original
    """)
    
    with engine.begin() as conn:
        conn.execute(update_export)
        conn.execute(update_prod)

# 6. Ejecución principal
def main():
    print("Cargando datos...")
    df_export, df_prod = load_data()
    
    print("Creando catálogo unificado...")
    catalog = create_unified_catalog(df_export, df_prod)
    
    # Guardar catálogo en SQL Server
    catalog.to_sql('catalogo_productos', engine, if_exists='replace', index=False)
    print("Catálogo de productos creado en la tabla [catalogo_productos]")
    
    # Añadir columnas a las tablas originales si no existen
    with engine.begin() as conn:
        # Para exportaciones
        try:
            conn.execute("""
                ALTER TABLE exportaciones_agricolas
                ADD id_producto INT NULL,
                    codigo_producto VARCHAR(8) NULL
            """)
        except:
            print("Las columnas ya existen en exportaciones_agricolas")
        
        # Para producción
        try:
            conn.execute("""
                ALTER TABLE produccion_agricola
                ADD id_producto INT NULL,
                    codigo_producto VARCHAR(8) NULL
            """)
        except:
            print("Las columnas ya existen en produccion_agricola")
    
    print("Actualizando tablas originales con los códigos...")
    update_original_tables(catalog, engine)
    print("Proceso completado exitosamente!")

if __name__ == "__main__":
    main()