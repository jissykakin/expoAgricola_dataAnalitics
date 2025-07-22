from .database import get_mysql_engine
import pandas as pd

def get_exportaciones():
    engine = get_mysql_engine()
    query = """
    SELECT Año, Mes, SUM([exportaciones_en_valor_(miles_usd_fob)]) as Total_Exportaciones
    FROM exportaciones
    GROUP BY Año, Mes
    """
    return pd.read_sql(query, engine)

def load_exportaciones():
    """Carga datos de exportaciones agrícolas"""
    engine = get_mysql_engine()
    query = """
    SELECT 
        año,
        mes,
        departamento,
        producto,        
        tradición_producto as tipo_producto,
        SUM([exportaciones_en_valor_(usd_fob)]) as valor_usd,
        SUM([exportaciones_en_volumen_(toneladas)]) as volumen_ton
    FROM [AgroAnalyticsDB].[dbo].[exportaciones_agricolas]
    GROUP BY año, mes, departamento, producto, tradición_producto
    """
    return pd.read_sql(query, engine)
    

def load_produccion_agricola():
    """Carga datos de produccion agrícolas"""
    engine = get_mysql_engine()
    query = """
        SELECT		
        p.codigo_dane_departamento
        ,d.departamento
        ,[desagregacion_cultivo]
        ,[cultivo]
        ,[ciclo_del_cultivo]
        ,[grupo_cultivo]
        ,[subgrupo]
        ,[ano]
        ,[periodo]
        ,[area_sembrada_(ha)]
        ,[area_cosechada_(ha)]
        ,[produccion_(t)]
        ,[rendimiento_(t/ha)]
        ,[nombre_cientifico_del_cultivo]
        ,[codigo_del_cultivo]
        ,[estado_fisico_del_cultivo]
        ,[id_producto]
        ,[codigo_producto]
    FROM produccion_agricola p 
    inner join departamentos d on p.codigo_dane_departamento = d.codigo_dane_departamento   
      
    """
    return pd.read_sql(query, engine)