# 

from dash import html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

def create_content():
    return dbc.Container([
        html.H2("Análisis Predictivo Avanzado", className="mb-4"),
        
        # Filtros y Configuración
        dbc.Card([
            dbc.CardHeader("Configuración del Modelo", className="bg-primary text-white"),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Label("Variable a Predecir"),
                        dcc.Dropdown(
                            id='predictive-target',
                            options=[
                                {'label': 'Valor (USD FOB)', 'value': 'valor'},
                                {'label': 'Volumen (Toneladas)', 'value': 'volumen'}
                            ],
                            value='valor'
                        )
                    ], md=3),
                    
                    dbc.Col([
                        html.Label("Grupo de Producto"),
                        dcc.Dropdown(
                            id='predictive-product-group',
                            options=[],
                            placeholder="Todos los grupos"
                        )
                    ], md=3),
                    
                    dbc.Col([
                        html.Label("Departamento"),
                        dcc.Dropdown(
                            id='predictive-department',
                            options=[],
                            placeholder="Todos los departamentos"
                        )
                    ], md=3),
                    
                    dbc.Col([
                        html.Label("Horizonte (meses)"),
                        dcc.Slider(
                            id='predictive-horizon',
                            min=3,
                            max=24,
                            step=3,
                            value=12,
                            marks={i: f'{i}' for i in range(3, 25, 3)}
                        )
                    ], md=3)
                ], className="mb-3"),
                
                dbc.Row([
                    dbc.Col([
                        html.Label("Modelo Predictivo"),
                        dcc.Dropdown(
                            id='predictive-model',
                            options=[
                                {'label': 'SARIMAX', 'value': 'sarimax'},
                                {'label': 'Prophet', 'value': 'prophet'},
                                {'label': 'Random Forest', 'value': 'rf'},
                                {'label': 'SVM', 'value': 'svm'}
                            ],
                            value='sarimax'
                        )
                    ], md=4),
                    
                    dbc.Col([
                        html.Label("Parámetros del Modelo"),
                        html.Div(id='model-params-container')
                    ], md=8)
                ]),
                
                # Contenedores ocultos para todos los parámetros posibles
                html.Div([
                    # Parámetros SARIMAX
                    html.Div([
                        dcc.Input(id='sarimax-p', type='number', min=0, max=3, value=1, style={'display': 'none'}),
                        dcc.Input(id='sarimax-d', type='number', min=0, max=2, value=1, style={'display': 'none'}),
                        dcc.Input(id='sarimax-q', type='number', min=0, max=3, value=1, style={'display': 'none'}),
                        dcc.Input(id='sarimax-P', type='number', min=0, max=3, value=1, style={'display': 'none'}),
                        dcc.Input(id='sarimax-D', type='number', min=0, max=2, value=1, style={'display': 'none'}),
                        dcc.Input(id='sarimax-Q', type='number', min=0, max=3, value=1, style={'display': 'none'}),
                        dcc.Input(id='sarimax-s', type='number', min=0, max=24, value=12, style={'display': 'none'}),
                        dcc.Dropdown(id='sarimax-trend', options=[], value='c', style={'display': 'none'})
                    ]),
                    
                     # Parámetros Prophet
                    html.Div([
                        dcc.Dropdown(id='prophet-yearly', options=[], value=True, style={'display': 'none'}),
                        html.Div(
                            dcc.Slider(id='prophet-interval', min=0.7, max=0.99, step=0.01, value=0.95, marks={}),
                            id='prophet-interval-container',
                            style={'display': 'none'}
                        ),
                        dcc.Dropdown(id='prophet-growth', options=[], value='linear', style={'display': 'none'})
                    ]),
                    
                    # Parámetros ML (Random Forest/SVM)
                    html.Div([
                        html.Div(
                            dcc.Slider(id='ml-test-size', min=10, max=50, step=5, value=20, marks={}),
                            id='ml-test-size-container',
                            style={'display': 'none'}
                        ),
                        dcc.Dropdown(id='ml-lags', options=[], value='default', style={'display': 'none'}),
                        dcc.Dropdown(id='ml-param', options=[], value='rbf', style={'display': 'none'})
                    ]),
                        ]),
                        
                        dbc.Row([
                            dbc.Col(
                                dbc.Button("Ejecutar Análisis", 
                                        id='run-analysis', 
                                        color="success", 
                                        className="w-100 mt-3"),
                                md=12
                            )
                        ])
                    ])
                ], className="mb-4"),
        
        # Resultados
        dbc.Tabs([
            dbc.Tab(label="Descomposición Temporal", tab_id="tab-decomposition"),
            dbc.Tab(label="Pruebas de Estacionariedad", tab_id="tab-stationarity"),
            dbc.Tab(label="Resultados del Modelo", tab_id="tab-results"),
            dbc.Tab(label="Comparación de Modelos", tab_id="tab-comparison")
        ], id="predictive-tabs", active_tab="tab-results"),
        
        html.Div(id="predictive-tab-content", className="p-3"),
        
        # Almacenes de datos
        dcc.Store(id='original-time-series'),
        dcc.Store(id='model-results'),
        dcc.Store(id='decomposition-figures'),
        dcc.Store(id='stationarity-tests')
    ], fluid=True)

content = create_content()