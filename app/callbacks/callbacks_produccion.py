from dash.dependencies import Input, Output, State
import pandas as pd
from analytics.produccion import AnalisisProduccion

def register_callbacks(app):
    @app.callback(
        [Output('filtro-ano', 'options'),
         Output('filtro-departamento', 'options'),
         Output('filtro-cultivo', 'options'),
         Output('filtro-grupo-cultivo', 'options'),
         Output('data-originales', 'data')],
        [Input('data-originales', 'modified_timestamp')],
        [State('data-originales', 'data')]
    )
    def cargar_datos(ts, datos_json):
        if datos_json is None:
            from utils.queries import load_produccion_agricola
            df = load_produccion_agricola()
            datos_json = df.to_json(date_format='iso', orient='split')
        else:
            df = pd.read_json(datos_json, orient='split')
        
        opciones_ano = [{'label': str(ano), 'value': ano} for ano in sorted(df['ano'].unique())]
        opciones_depto = [{'label': depto, 'value': depto} for depto in sorted(df['departamento'].unique())]
        opciones_cultivo = [{'label': cultivo, 'value': cultivo} for cultivo in sorted(df['cultivo'].unique())]
        opciones_grupo = [{'label': grupo, 'value': grupo} for grupo in sorted(df['grupo_cultivo'].unique())]
        
        return opciones_ano, opciones_depto, opciones_cultivo, opciones_grupo, datos_json
    
    @app.callback(
        Output('data-filtrados', 'data'),
        [Input('boton-filtrar', 'n_clicks')],
        [State('data-originales', 'data'),
         State('filtro-ano', 'value'),
         State('filtro-departamento', 'value'),
         State('filtro-cultivo', 'value'),
         State('filtro-grupo-cultivo', 'value'),
         State('filtro-tipo-periodo', 'value')]
    )
    def aplicar_filtros(n_clicks, datos_json, anos, departamentos, cultivos, grupos, tipo_periodo):
        if n_clicks is None:
            return datos_json
        
        df = pd.read_json(datos_json, orient='split')
        
        # Aplicar filtros
        if anos:
            df = df[df['ano'].isin(anos)]
        if departamentos:
            df = df[df['departamento'].isin(departamentos)]
        if cultivos:
            df = df[df['cultivo'].isin(cultivos)]
        if grupos:
            df = df[df['grupo_cultivo'].isin(grupos)]
        if tipo_periodo != 'Todos':
            df = df[df['tipo_periodo'] == tipo_periodo]
        
        return df.to_json(date_format='iso', orient='split')
    
    @app.callback(
        [Output('grafico-evolucion', 'figure'),
         Output('grafico-departamentos', 'figure'),
         Output('grafico-distribucion', 'figure'),
         Output('grafico-tendencia', 'figure'),
         Output('grafico-comparativo', 'figure')],  # Nuevo output para el gráfico comparativo
        [Input('data-filtrados', 'data')],
        [State('filtro-cultivo', 'value'),
         State('filtro-departamento', 'value'),
         State('filtro-grupo-cultivo', 'value')]  # Nuevo state para grupo de cultivo
    )
    def actualizar_graficos(datos_json, cultivos_seleccionados, departamento_seleccionado, grupo_cultivo):
        if datos_json is None:
            return {}, {}, {}, {}, {}  # Añadir un diccionario vacío para el nuevo gráfico
        
        df = pd.read_json(datos_json, orient='split')
        analisis = AnalisisProduccion(df)
        
        # Si hay múltiples seleccionados, tomamos el primero para algunos gráficos
        cultivo = cultivos_seleccionados[0] if cultivos_seleccionados and len(cultivos_seleccionados) == 1 else None
        departamento = departamento_seleccionado[0] if departamento_seleccionado and len(departamento_seleccionado) == 1 else None
        
        fig1 = analisis.grafico_evolucion_rendimiento(cultivos_seleccionados, departamento)
        fig2 = analisis.grafico_comparacion_departamentos(cultivo)
        fig3 = analisis.grafico_distribucion_cultivos()
        fig4 = analisis.grafico_tendencia_anual()
        
        # Nuevo gráfico comparativo
        fig5 = analisis.grafico_comparativo_rendimiento(
            top_n=10,
            departamento=departamento,
            grupo_cultivo=grupo_cultivo[0] if grupo_cultivo and len(grupo_cultivo) == 1 else None
        )
        
        return fig1, fig2, fig3, fig4, fig5


# from dash.dependencies import Input, Output, State
# import pandas as pd
# from analytics.produccion import AnalisisProduccion

# def register_callbacks(app):
#     @app.callback(
#         [Output('filtro-ano', 'options'),
#          Output('filtro-departamento', 'options'),
#          Output('filtro-cultivo', 'options'),
#          Output('filtro-grupo-cultivo', 'options'),
#          Output('data-originales', 'data')],
#         [Input('data-originales', 'modified_timestamp')],
#         [State('data-originales', 'data')]
#     )
#     def cargar_datos(ts, datos_json):
#         if datos_json is None:
#             from utils.queries import load_produccion_agricola
#             df = load_produccion_agricola()
#             datos_json = df.to_json(date_format='iso', orient='split')
#         else:
#             df = pd.read_json(datos_json, orient='split')
        
#         opciones_ano = [{'label': str(ano), 'value': ano} for ano in sorted(df['ano'].unique())]
#         opciones_depto = [{'label': depto, 'value': depto} for depto in sorted(df['departamento'].unique())]
#         opciones_cultivo = [{'label': cultivo, 'value': cultivo} for cultivo in sorted(df['cultivo'].unique())]
#         opciones_grupo = [{'label': grupo, 'value': grupo} for grupo in sorted(df['grupo_cultivo'].unique())]
        
#         return opciones_ano, opciones_depto, opciones_cultivo, opciones_grupo, datos_json
    
#     @app.callback(
#         Output('data-filtrados', 'data'),
#         [Input('boton-filtrar', 'n_clicks')],
#         [State('data-originales', 'data'),
#          State('filtro-ano', 'value'),
#          State('filtro-departamento', 'value'),
#          State('filtro-cultivo', 'value'),
#          State('filtro-grupo-cultivo', 'value'),
#          State('filtro-tipo-periodo', 'value')]
#     )
#     def aplicar_filtros(n_clicks, datos_json, anos, departamentos, cultivos, grupos, tipo_periodo):
#         if n_clicks is None:
#             return datos_json
        
#         df = pd.read_json(datos_json, orient='split')
        
#         # Aplicar filtros
#         if anos:
#             df = df[df['ano'].isin(anos)]
#         if departamentos:
#             df = df[df['departamento'].isin(departamentos)]
#         if cultivos:
#             df = df[df['cultivo'].isin(cultivos)]
#         if grupos:
#             df = df[df['grupo_cultivo'].isin(grupos)]
#         if tipo_periodo != 'Todos':
#             df = df[df['tipo_periodo'] == tipo_periodo]
        
#         return df.to_json(date_format='iso', orient='split')
    
#     @app.callback(
#         [Output('grafico-evolucion', 'figure'),
#          Output('grafico-departamentos', 'figure'),
#          Output('grafico-distribucion', 'figure'),
#          Output('grafico-tendencia', 'figure')],
#         [Input('data-filtrados', 'data')],
#         [State('filtro-cultivo', 'value'),
#          State('filtro-departamento', 'value')]
#     )
#     def actualizar_graficos(datos_json, cultivos_seleccionados, departamento_seleccionado):
#         if datos_json is None:
#             return {}, {}, {}, {}
        
#         df = pd.read_json(datos_json, orient='split')
#         analisis = AnalisisProduccion(df)
        
#         # Si hay múltiples seleccionados, tomamos el primero para algunos gráficos
#         cultivo = cultivos_seleccionados[0] if cultivos_seleccionados and len(cultivos_seleccionados) == 1 else None
#         departamento = departamento_seleccionado[0] if departamento_seleccionado and len(departamento_seleccionado) == 1 else None
        
#         fig1 = analisis.grafico_evolucion_rendimiento(cultivos_seleccionados, departamento)
#         fig2 = analisis.grafico_comparacion_departamentos(cultivo)
#         fig3 = analisis.grafico_distribucion_cultivos()
#         fig4 = analisis.grafico_tendencia_anual()
        
#         return fig1, fig2, fig3, fig4