from dash import callback, Input, Output, State, no_update, html, dcc
import pandas as pd
import dash_bootstrap_components as dbc
from utils.queries import load_produccion_agricola, load_exportaciones
from analytics.dashboard import DashboardAnalitico

def register_callbacks_dashboard(app):
    @app.callback(
        [Output('store-datos-produccion', 'data'),
        Output('store-datos-exportaciones', 'data')],
        [Input('trigger-load', 'data')],    
        prevent_initial_call=False
    )
    def cargar_datos(_):
        df_prod = load_produccion_agricola()
        df_exp = load_exportaciones()

        print(df_prod)
        return df_prod.to_json(date_format='iso', orient='split'), df_exp.to_json(date_format='iso', orient='split')
    
    @app.callback(
        [Output('metricas-produccion', 'children'),
         Output('metricas-exportaciones', 'children'),
         Output('grafico-resumen-produccion', 'figure'),
         Output('grafico-resumen-exportaciones', 'figure'),
         Output('grafico-top-cultivos', 'figure'),
         Output('grafico-top-exportaciones', 'figure')],
        [Input('store-datos-produccion', 'data'),
         Input('store-datos-exportaciones', 'data')]
    )
    def actualizar_dashboard(datos_prod_json, datos_exp_json):
        if not datos_prod_json or not datos_exp_json:
            return [], [], {}, {}, {}, {}
        
        df_prod = pd.read_json(datos_prod_json, orient='split')
        df_exp = pd.read_json(datos_exp_json, orient='split')
        
        analisis = DashboardAnalitico(df_prod, df_exp)
        metricas = analisis.generar_metricas()
        
        # Tarjetas de métricas para producción
        metricas_prod = [
            dbc.Card([
                dbc.CardBody([
                    html.H4(f"{metricas['produccion']['total_registros']:,}", className="card-title"),
                    html.P("Registros totales", className="card-text")
                ])
            ], className="text-center m-2"),
            dbc.Card([
                dbc.CardBody([
                    html.H4(f"{metricas['produccion']['total_cultivos']}", className="card-title"),
                    html.P("Cultivos diferentes", className="card-text")
                ])
            ], className="text-center m-2"),
            dbc.Card([
                dbc.CardBody([
                    html.H4(f"{metricas['produccion']['area_total_sembrada']:,.0f}", className="card-title"),
                    html.P("Hectáreas sembradas", className="card-text")
                ])
            ], className="text-center m-2"),
            dbc.Card([
                dbc.CardBody([
                    html.H4(f"{metricas['produccion']['produccion_total']:,.0f}", className="card-title"),
                    html.P("Toneladas producidas", className="card-text")
                ])
            ], className="text-center m-2")
        ]
        
        # Tarjetas de métricas para exportaciones
        metricas_exp = [
            dbc.Card([
                dbc.CardBody([
                    html.H4(f"{metricas['exportaciones']['total_registros']:,}", className="card-title"),
                    html.P("Registros totales", className="card-text")
                ])
            ], className="text-center m-2"),
            dbc.Card([
                dbc.CardBody([
                    html.H4(f"{metricas['exportaciones']['total_productos']}", className="card-title"),
                    html.P("Productos exportados", className="card-text")
                ])
            ], className="text-center m-2"),
            dbc.Card([
                dbc.CardBody([
                    html.H4(f"${metricas['exportaciones']['valor_total_exportado']:,.0f}", className="card-title"),
                    html.P("Valor total exportado", className="card-text")
                ])
            ], className="text-center m-2"),
            dbc.Card([
                dbc.CardBody([
                    html.H4(f"{metricas['exportaciones']['volumen_total']:,.0f}", className="card-title"),
                    html.P("Toneladas exportadas", className="card-text")
                ])
            ], className="text-center m-2")
        ]
        
        # Crear filas para las métricas
        metricas_produccion = dbc.Row([dbc.Col(m, md=3) for m in metricas_prod], className="mb-1")
        metricas_exportaciones = dbc.Row([dbc.Col(m, md=3) for m in metricas_exp], className="mb-1")
        
        # Generar gráficos
        fig_resumen_prod = analisis.grafico_resumen_produccion()
        fig_resumen_exp = analisis.grafico_resumen_exportaciones()
        fig_top_cultivos = analisis.grafico_top_cultivos()
        fig_top_export = analisis.grafico_top_exportaciones()
        
        return (
            metricas_produccion,
            metricas_exportaciones,
            fig_resumen_prod,
            fig_resumen_exp,
            fig_top_cultivos,
            fig_top_export
        )