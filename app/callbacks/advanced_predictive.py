# from dash import callback, Input, Output, State, no_update, html, dcc
# import dash_bootstrap_components as dbc
# import pandas as pd
# import json
# from analytics.advanced_predictive import AdvancedPredictiveModels
# from utils.queries import load_exportaciones

# def register_advanced_predictive_callbacks(app):
#     @app.callback(
#         Output('model-params-container', 'children'),
#         Output('sarimax-p', 'style'),
#         Output('sarimax-d', 'style'),
#         Output('sarimax-q', 'style'),
#         Output('sarimax-P', 'style'),
#         Output('sarimax-D', 'style'),
#         Output('sarimax-Q', 'style'),
#         Output('sarimax-s', 'style'),
#         Output('sarimax-trend', 'style'),
#         Output('prophet-yearly', 'style'),
#         Output('prophet-interval-container', 'style'),
#         Output('prophet-growth', 'style'),
#         Output('ml-test-size-container', 'style'),
#         Output('ml-lags', 'style'),
#         Output('ml-param', 'style'),
#         Input('predictive-model', 'value')
#     )
#     def update_model_params(model_type):
#         # Estilos base (ocultar todos)
#         base_style = {'display': 'none'}
#         styles = {
#             'sarimax-p': base_style,
#             'sarimax-d': base_style,
#             'sarimax-q': base_style,
#             'sarimax-P': base_style,
#             'sarimax-D': base_style,
#             'sarimax-Q': base_style,
#             'sarimax-s': base_style,
#             'sarimax-trend': base_style,
#             'prophet-yearly': base_style,
#             'prophet-interval-container': base_style,
#             'prophet-growth': base_style,
#             'ml-test-size-container': base_style,
#             'ml-lags': base_style,
#             'ml-param': base_style
#         }
        
#         if model_type == 'sarimax':
#             content = dbc.Row([
#                 # ... (contenido SARIMAX)
#             ])
#             # Actualizar estilos SARIMAX
#             for key in ['sarimax-p', 'sarimax-d', 'sarimax-q', 'sarimax-P', 
#                        'sarimax-D', 'sarimax-Q', 'sarimax-s', 'sarimax-trend']:
#                 styles[key] = {'display': 'block'}
        
#         elif model_type == 'prophet':
#             content = dbc.Row([
#                 dbc.Col([
#                     html.Label("Estacionalidad Anual"),
#                     dcc.Dropdown(
#                         id='prophet-yearly',
#                         options=[
#                             {'label': 'Sí', 'value': True},
#                             {'label': 'No', 'value': False}
#                         ],
#                         value=True
#                     )
#                 ], md=4),
                
#                 dbc.Col([
#                     html.Label("Intervalo de Incertidumbre"),
#                     dcc.Slider(
#                         id='prophet-interval',
#                         min=0.7,
#                         max=0.99,
#                         step=0.01,
#                         value=0.95,
#                         marks={0.8: '80%', 0.9: '90%', 0.95: '95%'}
#                     )
#                 ], md=4),
                
#                 dbc.Col([
#                     html.Label("Modo de Crecimiento"),
#                     dcc.Dropdown(
#                         id='prophet-growth',
#                         options=[
#                             {'label': 'Lineal', 'value': 'linear'},
#                             {'label': 'Logístico', 'value': 'logistic'}
#                         ],
#                         value='linear'
#                     )
#                 ], md=4)
#             ])
#             styles.update({
#                 'prophet-yearly': {'display': 'block'},
#                 'prophet-interval-container': {'display': 'block'},
#                 'prophet-growth': {'display': 'block'}
#             })
        
#         elif model_type in ['rf', 'svm']:
#             content = dbc.Row([
#                 dbc.Col([
#                     html.Label("Tamaño de Test (%)"),
#                     dcc.Slider(
#                         id='ml-test-size',
#                         min=10,
#                         max=50,
#                         step=5,
#                         value=20,
#                         marks={i: f'{i}%' for i in range(10, 51, 10)}
#                     )
#                 ], md=4),
                
#                 dbc.Col([
#                     html.Label("Número de Lags"),
#                     dcc.Dropdown(
#                         id='ml-lags',
#                         options=[
#                             {'label': '1, 2, 3, 12', 'value': 'default'},
#                             {'label': '1-6 meses', 'value': 'short'},
#                             {'label': '1-12 meses', 'value': 'medium'},
#                             {'label': '1-24 meses', 'value': 'long'}
#                         ],
#                         value='default',
#                         multi=False
#                     )
#                 ], md=4),
                
#                 dbc.Col([
#                     html.Label("Kernel (SVM)" if model_type == 'svm' else "Árboles (RF)"),
#                     dcc.Dropdown(
#                         id='ml-param',
#                         options=[
#                             {'label': 'RBF', 'value': 'rbf'} if model_type == 'svm' else {'label': '100 árboles', 'value': 100},
#                             {'label': 'Lineal', 'value': 'linear'} if model_type == 'svm' else {'label': '200 árboles', 'value': 200},
#                             {'label': 'Polinomial', 'value': 'poly'} if model_type == 'svm' else {'label': '500 árboles', 'value': 500}
#                         ],
#                         value='rbf' if model_type == 'svm' else 100
#                     )
#                 ], md=4)
#             ])
#             styles.update({
#                 'ml-test-size-container': {'display': 'block'},
#                 'ml-lags': {'display': 'block'},
#                 'ml-param': {'display': 'block'}
#             })
        
#         else:
#             content = html.Div("Seleccione un modelo")
        
#         return (
#             content,
#             styles['sarimax-p'],
#             styles['sarimax-d'],
#             styles['sarimax-q'],
#             styles['sarimax-P'],
#             styles['sarimax-D'],
#             styles['sarimax-Q'],
#             styles['sarimax-s'],
#             styles['sarimax-trend'],
#             styles['prophet-yearly'],
#             styles['prophet-interval-container'],
#             styles['prophet-growth'],
#             styles['ml-test-size-container'],
#             styles['ml-lags'],
#             styles['ml-param']
#         )
    
#     # Callback para ejecutar el análisis (simplificado)
#     @app.callback(
#         Output('model-results', 'data'),
#         Input('run-analysis', 'n_clicks'),
#         State('predictive-model', 'value'),
#         State('predictive-target', 'value'),
#         State('sarimax-p', 'value'),
#         State('sarimax-d', 'value'),
#         State('sarimax-q', 'value'),
#         State('sarimax-P', 'value'),
#         State('sarimax-D', 'value'),
#         State('sarimax-Q', 'value'),
#         State('sarimax-s', 'value'),
#         State('sarimax-trend', 'value'),
#         State('prophet-yearly', 'value'),
#         State('prophet-interval-container', 'value'),
#         State('prophet-growth', 'value'),
#         State('ml-test-size', 'value'),
#         State('ml-lags', 'value'),
#         State('ml-param', 'value'),
#         prevent_initial_call=True
#     )
#     def run_analysis(n_clicks, model_type, target, *args):
#         if n_clicks is None:
#             return no_update
        
#         # Determinar qué parámetros usar según el modelo
#         if model_type == 'sarimax':
#             p, d, q, P, D, Q, s, trend = args[:8]
#             print(f"Ejecutando SARIMAX con: p={p}, d={d}, q={q}, P={P}, D={D}, Q={Q}, s={s}, trend={trend}")
#             # Aquí iría tu lógica para SARIMAX
        
#         elif model_type == 'prophet':
#             yearly, interval, growth = args[8:11]
#             print(f"Ejecutando Prophet con: yearly={yearly}, interval={interval}, growth={growth}")
#             # Aquí iría tu lógica para Prophet
        
#         elif model_type in ['rf', 'svm']:
#             test_size, lags, param = args[11:14]
#             print(f"Ejecutando {model_type.upper()} con: test_size={test_size}, lags={lags}, param={param}")
#             # Aquí iría tu lógica para ML models
        
#         return {'status': 'success', 'model': model_type}




from dash import callback, Input, Output, State, no_update, html, dcc
import dash_bootstrap_components as dbc
import pandas as pd
import json
from analytics.advanced_predictive import AdvancedPredictiveModels
from utils.queries import load_exportaciones

def register_advanced_predictive_callbacks(app):
    # Cargar opciones de filtros
    @app.callback(
        Output('predictive-product-group', 'options'),
        Output('predictive-department', 'options'),
        Input('original-time-series', 'data')
    )
    def load_filter_options(ts_data):
        if ts_data is None:
            return [], []
            
        df = pd.read_json(ts_data, orient='split')
        grupos = [{'label': g, 'value': g} for g in sorted(df['tradición_producto'].unique())]
        deptos = [{'label': d, 'value': d} for d in sorted(df['departamento'].unique())]
        return grupos, deptos
    
    # Actualizar parámetros del modelo según selección
    @app.callback(
        Output('model-params-container', 'children'),
        Input('predictive-model', 'value')
    )
    def update_model_params(model_type):
        if model_type == 'sarimax':
            return dbc.Row([
                dbc.Col([
                    html.Label("Orden (p,d,q)"),
                    dcc.Input(id='sarimax-p', type='number', min=0, max=3, value=1),
                    dcc.Input(id='sarimax-d', type='number', min=0, max=2, value=1),
                    dcc.Input(id='sarimax-q', type='number', min=0, max=3, value=1)
                ], md=4),
                
                dbc.Col([
                    html.Label("Orden Estacional (P,D,Q,s)"),
                    dcc.Input(id='sarimax-P', type='number', min=0, max=3, value=1),
                    dcc.Input(id='sarimax-D', type='number', min=0, max=2, value=1),
                    dcc.Input(id='sarimax-Q', type='number', min=0, max=3, value=1),
                    dcc.Input(id='sarimax-s', type='number', min=0, max=24, value=12)
                ], md=4),
                
                dbc.Col([
                    html.Label("Tendencia"),
                    dcc.Dropdown(
                        id='sarimax-trend',
                        options=[
                            {'label': 'Ninguna', 'value': 'n'},
                            {'label': 'Constante', 'value': 'c'},
                            {'label': 'Lineal', 'value': 't'},
                            {'label': 'Cuadrática', 'value': 'ct'}
                        ],
                        value='c'
                    )
                ], md=4)
            ])
        
        elif model_type == 'prophet':
            return dbc.Row([
                dbc.Col([
                    html.Label("Estacionalidad Anual"),
                    dcc.Dropdown(
                        id='prophet-yearly',
                        options=[
                            {'label': 'Sí', 'value': True},
                            {'label': 'No', 'value': False}
                        ],
                        value=True
                    )
                ], md=4),
                
                dbc.Col([
                    html.Label("Intervalo de Incertidumbre"),
                    dcc.Slider(
                        id='prophet-interval',
                        min=0.7,
                        max=0.99,
                        step=0.01,
                        value=0.95,
                        marks={0.8: '80%', 0.9: '90%', 0.95: '95%'}
                    )
                ], md=4),
                
                dbc.Col([
                    html.Label("Modo de Crecimiento"),
                    dcc.Dropdown(
                        id='prophet-growth',
                        options=[
                            {'label': 'Lineal', 'value': 'linear'},
                            {'label': 'Logístico', 'value': 'logistic'}
                        ],
                        value='linear'
                    )
                ], md=4)
            ])
        
        elif model_type in ['rf', 'svm']:
            return dbc.Row([
                dbc.Col([
                    html.Label("Tamaño de Test (%)"),
                    dcc.Slider(
                        id='ml-test-size',
                        min=10,
                        max=50,
                        step=5,
                        value=20,
                        marks={i: f'{i}%' for i in range(10, 51, 10)}
                    )
                ], md=4),
                
                dbc.Col([
                    html.Label("Número de Lags"),
                    dcc.Dropdown(
                        id='ml-lags',
                        options=[
                            {'label': '1, 2, 3, 12', 'value': 'default'},
                            {'label': '1-6 meses', 'value': 'short'},
                            {'label': '1-12 meses', 'value': 'medium'},
                            {'label': '1-24 meses', 'value': 'long'}
                        ],
                        value='default',
                        multi=False
                    )
                ], md=4),
                
                dbc.Col([
                    html.Label("Kernel (SVM)" if model_type == 'svm' else "Árboles (RF)"),
                    dcc.Dropdown(
                        id='ml-param',
                        options=[
                            {'label': 'RBF', 'value': 'rbf'} if model_type == 'svm' else {'label': '100 árboles', 'value': 100},
                            {'label': 'Lineal', 'value': 'linear'} if model_type == 'svm' else {'label': '200 árboles', 'value': 200},
                            {'label': 'Polinomial', 'value': 'poly'} if model_type == 'svm' else {'label': '500 árboles', 'value': 500}
                        ],
                        value='rbf' if model_type == 'svm' else 100
                    )
                ], md=4)
            ])
        
        return html.Div("Seleccione un modelo")
    
    # Ejecutar análisis completo
    @app.callback(
        Output('original-time-series', 'data'),
        Output('decomposition-figures', 'data'),
        Output('stationarity-tests', 'data'),
        Input('run-analysis', 'n_clicks'),
        State('predictive-target', 'value'),
        State('predictive-product-group', 'value'),
        State('predictive-department', 'value'),
        prevent_initial_call=True
    )
    def run_initial_analysis(n_clicks, target, product_group, department):
        if n_clicks is None:
            return no_update, no_update, no_update
        
        # Cargar datos (aquí debes implementar tu propia carga de datos)
        df = load_exportaciones()  # Reemplaza con tu función
        
        print("Primer df",df.head())
        # Inicializar modelo avanzado
        predictor = AdvancedPredictiveModels(df)
        
        # Preparar serie temporal
        ts = predictor.prepare_time_series(
            grupo=product_group,
            depto=department
        )
        
        # Descomposición estacional
        decomp_fig = predictor.seasonal_decomposition(ts, column=target)
        
        # Pruebas de estacionariedad
        stationarity = predictor.test_stationarity(ts, column=target)
        
        # Serializar resultados
        ts_json = ts.reset_index().to_json(date_format='iso', orient='split')
        decomp_json = json.dumps({'decomposition': 'figure'})  # Serializar figura matplotlib
        stationarity_json = json.dumps(stationarity)
        
        return ts_json, decomp_json, stationarity_json
    
    # Ejecutar modelo predictivo seleccionado
    @app.callback(
        Output('model-results', 'data'),
        Input('run-analysis', 'n_clicks'),
        State('original-time-series', 'data'),
        State('predictive-target', 'value'),
        State('predictive-model', 'value'),
        State('predictive-horizon', 'value'),
        State('sarimax-p', 'value'),
        State('sarimax-d', 'value'),
        State('sarimax-q', 'value'),
        State('sarimax-P', 'value'),
        State('sarimax-D', 'value'),
        State('sarimax-Q', 'value'),
        State('sarimax-s', 'value'),
        State('sarimax-trend', 'value'),
        State('prophet-yearly', 'value'),
        State('prophet-interval', 'value'),
        State('prophet-growth', 'value'),
        State('ml-test-size', 'value'),
        State('ml-lags', 'value'),
        State('ml-param', 'value'),
        prevent_initial_call=True
    )
    def run_predictive_model(n_clicks, ts_data, target, model_type, horizonte, 
                           p, d, q, P, D, Q, s, trend,
                           yearly, interval, growth,
                           test_size, lags, ml_param):
        if n_clicks is None or ts_data is None:
            return no_update
            
        ts = pd.read_json(ts_data, orient='split').set_index('fecha')
        predictor = AdvancedPredictiveModels(pd.DataFrame())  # Solo necesitamos los métodos
        
        results = {}
        
        if model_type == 'sarimax':
            fig, summary = predictor.model_sarimax(
                ts, 
                column=target,
                order=(p, d, q),
                seasonal_order=(P, D, Q, s),
                steps=horizonte
            )
            results = {
                'figure': fig.to_dict(),
                'summary': str(summary),
                'type': 'sarimax'
            }
        
        elif model_type == 'prophet':
            fig, forecast = predictor.model_prophet(
                ts,
                column=target,
                periods=horizonte
            )
            results = {
                'figure': fig.to_dict(),
                'forecast': forecast.to_dict('records'),
                'type': 'prophet'
            }
        
        elif model_type == 'rf':
            fig, metrics = predictor.model_random_forest(
                ts,
                column=target,
                test_size=test_size/100,
                n_estimators=ml_param
            )
            results = {
                'figure': fig.to_dict(),
                'metrics': metrics,
                'type': 'rf'
            }
        
        elif model_type == 'svm':
            fig, metrics = predictor.model_svm(
                ts,
                column=target,
                test_size=test_size/100,
                kernel=ml_param
            )
            results = {
                'figure': fig.to_dict(),
                'metrics': metrics,
                'type': 'svm'
            }
        
        return json.dumps(results)
    
    # Mostrar contenido según pestaña seleccionada
    @app.callback(
        Output('predictive-tab-content', 'children'),
        Input('predictive-tabs', 'active_tab'),
        State('decomposition-figures', 'data'),
        State('stationarity-tests', 'data'),
        State('model-results', 'data')
    )
    def render_tab_content(active_tab, decomp_data, stationarity_data, model_data):
        if active_tab == "tab-decomposition":
            if decomp_data is None:
                return dbc.Alert("Ejecute el análisis primero", color="warning")
            
            # Aquí deberías implementar la visualización de la descomposición
            return html.Div([
                html.H4("Descomposición Temporal"),
                html.Img(src='/assets/decomposition.png')  # Ejemplo, ajusta según tu implementación
            ])
        
        elif active_tab == "tab-stationarity":
            if stationarity_data is None:
                return dbc.Alert("Ejecute el análisis primero", color="warning")
                
            tests = json.loads(stationarity_data)
            return dbc.Table([
                html.Thead(html.Tr([html.Th("Prueba"), html.Th("Estadístico"), html.Th("p-valor")])),
                html.Tbody([
                    html.Tr([html.Td("KPSS"), html.Td(f"{tests['KPSS']['Statistic']:.4f}"), html.Td(f"{tests['KPSS']['p-value']:.4f}")]),
                    html.Tr([html.Td("ADF"), html.Td(f"{tests['ADF']['Statistic']:.4f}"), html.Td(f"{tests['ADF']['p-value']:.4f}")])
                ])
            ], bordered=True)
        
        elif active_tab == "tab-results":
            if model_data is None:
                return dbc.Alert("Seleccione y ejecute un modelo", color="warning")
                
            results = json.loads(model_data)
            return html.Div([
                dcc.Graph(figure=results['figure']),
                html.Hr(),
                html.H4("Resultados del Modelo"),
                html.Pre(results.get('summary', str(results.get('metrics', ''))))
            ])
        
        elif active_tab == "tab-comparison":
            return html.Div([
                html.H4("Comparación de Modelos"),
                dbc.Alert("Esta funcionalidad está en desarrollo", color="info")
            ])
        
        return html.Div()