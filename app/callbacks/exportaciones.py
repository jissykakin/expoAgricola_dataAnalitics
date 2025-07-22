from dash import callback, Input, Output, State, no_update
import pandas as pd
from analytics.exportaciones import AnalisisExportaciones
import plotly.express as px

def register_exportaciones_callbacks(app):
    @app.callback(
        Output('datos-filtrados', 'data'),
        Input('boton-filtrar', 'n_clicks'),
        State('filtro-anos', 'value'),
        State('filtro-tipo-producto', 'value'),
        State('filtro-departamento', 'value'),
        State('datos-originales', 'data'),
        prevent_initial_call=True
    )
    def filtrar_datos(n_clicks, rango_anos, tipos_producto, departamentos, datos_originales):
        # Verificar si el callback fue disparado por el botón
        if n_clicks is None:
            return no_update
            
        # Verificar si hay valores vacíos
        if not tipos_producto or not departamentos:
            # Alternativa 1: Devolver datos vacíos
            empty_df = pd.DataFrame(columns=['año', 'mes', 'valor_usd', 'tipo_producto', 'departamento'])
            return empty_df.to_json(date_format='iso', orient='split')
            
            # Alternativa 2: Mantener los datos anteriores
            # return no_update
            
        # Convertir datos originales a DataFrame
        df = pd.read_json(datos_originales, orient='split')
        
        # Aplicar filtros
        df_filtrado = df[
            (df['año'] >= rango_anos[0]) & 
            (df['año'] <= rango_anos[1]) & 
            (df['tipo_producto'].isin(tipos_producto)) & 
            (df['departamento'].isin(departamentos))
        ]
        
        return df_filtrado.to_json(date_format='iso', orient='split')

    @app.callback(
        Output('grafico-anual', 'figure'),
        Output('grafico-mensual', 'figure'),
        Output('grafico-tipo', 'figure'),
        Output('grafico-depto', 'figure'),
        Input('datos-filtrados', 'data'),
        prevent_initial_call=True
    )
    def actualizar_graficos(datos_filtrados):
        """Actualiza todos los gráficos con los datos filtrados"""
        # Convertir datos filtrados a DataFrame
        df_filtrado = pd.read_json(datos_filtrados, orient='split')
        analisis = AnalisisExportaciones(df_filtrado)
        
        # Generar gráficos con los datos filtrados
        fig_anual = analisis.exportaciones_anuales()
        fig_mensual = analisis.evolucion_mensual()
        fig_tipo = analisis.exportaciones_por_tipo()
        fig_depto = analisis.exportaciones_por_departamento()
        
        return fig_anual, fig_mensual, fig_tipo, fig_depto

    @app.callback(
        Output('grafico-anual', 'figure', allow_duplicate=True),
        Output('grafico-mensual', 'figure', allow_duplicate=True),
        Output('grafico-tipo', 'figure', allow_duplicate=True),
        Output('grafico-depto', 'figure', allow_duplicate=True),
        Input('url', 'pathname'),
        State('datos-filtrados', 'data'),
        prevent_initial_call=True
    )
    def inicializar_graficos(pathname, datos_almacenados):
        """Inicializa los gráficos cuando se carga la página"""
        if pathname == '/exportaciones/descriptivo' or pathname == '/':
            df = pd.read_json(datos_almacenados, orient='split')
            analisis = AnalisisExportaciones(df)
            
            fig_anual = analisis.exportaciones_anuales()
            fig_mensual = analisis.evolucion_mensual()
            fig_tipo = analisis.exportaciones_por_tipo()
            fig_depto = analisis.exportaciones_por_departamento()
            
            return fig_anual, fig_mensual, fig_tipo, fig_depto
        
        # Retornar figuras vacías si no es la ruta correcta
        empty_fig = px.scatter()
        empty_fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        return empty_fig, empty_fig, empty_fig, empty_fig
    
    @app.callback(
        Output('productos_animados', 'figure'),
        Input('datos-filtrados', 'data'),
        Input('select-numero-productos', 'value'),
        prevent_initial_call=True
    )
    def actualizar_evolucion_productos(datos_filtrados, top_n):
        """Actualiza el gráfico de evolución de productos"""
        df = pd.read_json(datos_filtrados, orient='split')
        analisis = AnalisisExportaciones(df)
        return analisis.productos_animados(top_n=top_n)