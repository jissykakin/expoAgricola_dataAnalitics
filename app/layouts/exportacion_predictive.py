from dash import html, dcc
import dash_bootstrap_components as dbc
from utils.queries import load_exportaciones
from analytics.predictive_exportaciones import PredictiveExportaciones
from dash.dependencies import Input, Output, State

def create_content():
    try:
        # Cargar datos
        df = load_exportaciones()
        if df.empty:
            return dbc.Alert("No hay datos disponibles", color="danger")
            
        # Obtener opciones para filtros
        años = sorted(df['año'].unique())
        productos = sorted(df['producto'].unique())
        departamentos = sorted(df['departamento'].unique())
        
        # Componentes del layout
        return dbc.Container([
            html.H2("Análisis Predictivo de Exportaciones", className="mb-4 text-4xl font-bold"),
            
            # Filtros
            dbc.Card([
                dbc.CardHeader("Filtros Predictivos", className="bg-primary text-white"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Producto"),
                            dcc.Dropdown(
                                id='filtro-producto-predict',
                                options=[{'label': p, 'value': p} for p in productos],
                                value=None,
                                multi=False,
                                placeholder="Seleccione un producto"
                            )
                        ], md=4),
                        
                        dbc.Col([
                            html.Label("Departamento"),
                            dcc.Dropdown(
                                id='filtro-depto-predict',
                                options=[{'label': d, 'value': d} for d in departamentos],
                                value=None,
                                multi=False,
                                placeholder="Todos los departamentos"
                            )
                        ], md=4),
                        
                        dbc.Col([
                            html.Label("Modelo Predictivo"),
                            dcc.Dropdown(
                                id='filtro-modelo',
                                options=[
                                    {'label': 'SARIMA', 'value': 'sarima'},
                                    {'label': 'Prophet', 'value': 'prophet'},
                                    {'label': 'Random Forest', 'value': 'rf'}
                                ],
                                value='arima'
                            )
                        ], md=4)
                    ], className="mb-3"),
                    
                    dbc.Row([
                        dbc.Col([
                            html.Label("Horizonte de Predicción (meses)"),
                            dcc.Slider(
                                id='filtro-horizonte',
                                min=3,
                                max=24,
                                step=3,
                                value=12,
                                marks={i: f'{i}' for i in range(3, 25, 3)}
                            )
                        ], md=12)
                    ]),
                    
                    dbc.Row([
                        dbc.Col(
                            dbc.Button("Generar Predicción", 
                                     id="boton-predict", 
                                     color="success", 
                                     className="w-100 mt-3"),
                            md=12
                        )
                    ])
                ])
            ], className="mb-4"),
            
            # Resultados
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Predicción de Valor (USD FOB)", className="bg-info text-white"),
                        dbc.CardBody(
                            dcc.Loading(
                                dcc.Graph(id='grafico-prediccion-valor'),
                                type="cube"
                            )
                        )
                    ])
                ], md=6),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Predicción de Volumen (Toneladas)", className="bg-info text-white"),
                        dbc.CardBody(
                            dcc.Loading(
                                dcc.Graph(id='grafico-prediccion-volumen'),
                                type="cube"
                            )
                        )
                    ])
                ], md=6)
            ], className="mb-4"),
            
            # Métricas de evaluación
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Métricas del Modelo", className="bg-secondary text-white"),
                        dbc.CardBody(id='metricas-modelo')
                    ])
                ], md=12)
            ]),
            
            # Almacenes de datos
            dcc.Store(id='datos-originales', data=df.to_json(date_format='iso', orient='split')),
            dcc.Store(id='datos-entrenamiento', data=None),
            dcc.Store(id='resultados-prediccion', data=None)
        ], fluid=True)
        
    except Exception as e:
        print(f"Error en layout predictivo: {str(e)}")
        return dbc.Alert(f"Error al cargar datos: {str(e)}", color="danger")

content = create_content()