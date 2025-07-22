from dash import callback, Input, Output, State, no_update, html
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import json
from analytics.predictive_exportaciones import PredictiveExportaciones

def register_predictive_callbacks(app):
    @app.callback(
    Output('resultados-prediccion', 'data'),
    Output('datos-entrenamiento', 'data'),
    Input('boton-predict', 'n_clicks'),
    State('datos-originales', 'data'),
    State('filtro-producto-predict', 'value'),
    State('filtro-depto-predict', 'value'),
    State('filtro-modelo', 'value'),
    State('filtro-horizonte', 'value'),
    prevent_initial_call=True
)
   
    def generar_prediccion(n_clicks, datos_originales, producto, departamento, modelo, horizonte):
        if n_clicks is None:
            return no_update, no_update
            
        df = pd.read_json(datos_originales, orient='split')
        analisis = PredictiveExportaciones(df)
        
        resultados = analisis.predecir(
            producto=producto,
            departamento=departamento,
            modelo=modelo,
            horizonte=horizonte
        )
        
        # Convertir DataFrames a diccionarios serializables
        resultados_serializables = {
            'valor': {
                'predicciones': resultados['valor']['predicciones'].to_dict('records'),
                'metricas': resultados['valor']['metricas']
            },
            'volumen': {
                'predicciones': resultados['volumen']['predicciones'].to_dict('records'),
                'metricas': resultados['volumen']['metricas']
            },
            'historico': resultados['historico'].to_dict('records')
        }

          
        return resultados_serializables, datos_originales


    @app.callback(
    Output('grafico-prediccion-valor', 'figure'),
    Output('grafico-prediccion-volumen', 'figure'),
    Output('metricas-modelo', 'children'),
    Input('resultados-prediccion', 'data'),
    State('datos-entrenamiento', 'data'),
    prevent_initial_call=True
)
    def actualizar_graficos(resultados_serializables, datos_entrenamiento):
        if not resultados_serializables:
            return no_update, no_update, no_update
            
        # Convertir de vuelta a DataFrames
        resultados = {
            'valor': {
                'predicciones': pd.DataFrame(resultados_serializables['valor']['predicciones']),
                'metricas': resultados_serializables['valor']['metricas']
            },
            'volumen': {
                'predicciones': pd.DataFrame(resultados_serializables['volumen']['predicciones']),
                'metricas': resultados_serializables['volumen']['metricas']
            },
            'historico': pd.DataFrame(resultados_serializables['historico'])
        }
        
        df_entrenamiento = pd.read_json(datos_entrenamiento, orient='split')
        analisis = PredictiveExportaciones(df_entrenamiento)
        
        # Gráfico de valor
        try:
            fig_valor = analisis.generar_grafico_prediccion(
                df_historico=pd.DataFrame(resultados['historico']),
                df_prediccion=pd.DataFrame(resultados['valor']['predicciones']),
                columna='valor_usd',
                titulo='Predicción de Valor de Exportaciones (USD FOB)'
            )
        except Exception as e:
            fig_valor = analisis._create_empty_figure(f"Error en valor: {str(e)}")
        
        # Gráfico de volumen
        try:
            fig_volumen = analisis.generar_grafico_prediccion(
                df_historico=pd.DataFrame(resultados['historico']),
                df_prediccion=pd.DataFrame(resultados['volumen']['predicciones']),
                columna='volumen_ton',
                titulo='Predicción de Volumen de Exportaciones (Toneladas)'
            )
        except Exception as e:
            fig_volumen = analisis._create_empty_figure(f"Error en volumen: {str(e)}")
        
       
        # 3. Mostrar diagnósticos
        try:
            fig_diag = analisis.mostrar_diagnosticos_sarima(resultados, 'diagnosticos')
            fig_diag.show()
        except Exception as e:
            fig_diag = analisis._create_empty_figure(f"Error en diagnostico: {str(e)}")
        
        # Métricas con manejo de errores
        metricas = []
        try:
            if 'error' in resultados['valor']['metricas']:
                metricas.append(dbc.Alert(
                    f"Error en modelo de valor: {resultados['valor']['metricas']['error']}",
                    color="danger"
                ))
            elif resultados['valor']['metricas']:
                metricas.append(html.H5("Métricas para Valor (USD FOB)"))
                metricas.append(html.P(f"MAE: {resultados['valor']['metricas'].get('mae', 'N/A'):,.2f}"))
                metricas.append(html.P(f"RMSE: {resultados['valor']['metricas'].get('rmse', 'N/A'):,.2f}"))
                if 'error_pct' in resultados['valor']['metricas']:
                    metricas.append(html.P(f"Error %: {resultados['valor']['metricas']['error_pct']:.2f}%"))
        except Exception as e:
            metricas.append(dbc.Alert(f"Error al calcular métricas: {str(e)}", color="danger"))
        
        # Similar para volumen...
        
        return fig_valor, fig_volumen, metricas
    
    