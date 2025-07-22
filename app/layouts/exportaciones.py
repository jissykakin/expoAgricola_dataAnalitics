from dash import html, dcc
import dash_bootstrap_components as dbc
from utils.queries import load_exportaciones
from analytics.exportaciones import AnalisisExportaciones
from dash.dependencies import Input, Output, State

def create_content():
    try:
        # Cargar y validar datos
        df = load_exportaciones()
        if df.empty:
            return dbc.Alert("No hay datos disponibles", color="danger")
            
        # Obtener opciones únicas para los filtros
        años = sorted(df['año'].unique())
        tipos_producto = sorted(df['tipo_producto'].unique())
        departamentos = sorted(df['departamento'].unique())
        
        # Crear controles de filtro
        filtros = dbc.Card([
            dbc.CardHeader("Filtros", className="bg-primary text-white"),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Label("Rango de Años"),
                        dcc.RangeSlider(
                            id='filtro-anos',
                            min=min(años),
                            max=max(años),
                            step=1,
                            value=[min(años), max(años)],
                            marks={str(año): str(año) for año in años},
                            tooltip={"placement": "bottom", "always_visible": True}
                        )
                    ], md=12),
                ]),
                
                dbc.Row([
                    dbc.Col([
                        html.Label("Tipo de Producto"),
                        dcc.Dropdown(
                            id='filtro-tipo-producto',
                            options=[{'label': tipo, 'value': tipo} for tipo in tipos_producto],
                            value=tipos_producto,
                            multi=True,
                            placeholder="Seleccione tipos de producto"
                        )
                    ], md=6),
                    
                    dbc.Col([
                        html.Label("Departamento"),
                        dcc.Dropdown(
                            id='filtro-departamento',
                            options=[{'label': depto, 'value': depto} for depto in departamentos],
                            value=departamentos,
                            multi=True,
                            placeholder="Seleccione departamentos"
                        )
                    ], md=6),
                ], className="mt-3"),
                
                dbc.Row([
                    dbc.Col(
                        dbc.Button("Aplicar Filtros", id="boton-filtrar", color="primary", className="w-100 mt-3"),
                        md=12
                    )
                ])
            ])
        ], className="mb-4")
        
        return dbc.Container([
            html.H2("Análisis Descriptivo de Exportaciones", className="mb-4 text-4xl font-bold"),
            
            # Sección de filtros
            filtros,
            
            # Gráficos con loading components para mejor UX
            dbc.Row([
                dbc.Col(dcc.Loading(
                    dcc.Graph(
                        id='grafico-anual',
                        config={'displayModeBar': True}
                    ), type="graph"), md=6),
                
                dbc.Col(dcc.Loading(
                    dcc.Graph(
                        id='grafico-mensual',
                        config={'displayModeBar': True}
                    ), type="graph"), md=6),
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col(dcc.Loading(
                    dcc.Graph(
                        id='grafico-tipo',
                        config={'displayModeBar': True}
                    ), type="graph"), md=6),
                
                dbc.Col(dcc.Loading(
                    dcc.Graph(
                        id='grafico-depto',
                        config={'displayModeBar': True}
                    ), type="graph"), md=6),
            ], className="mb-4"),

             dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(
                            html.Div([
                                "Evolución Anual de Productos Más Exportados",
                                dbc.Select(
                                    id='select-numero-productos',
                                    options=[
                                        {'label': 'Top 5', 'value': 5},
                                        {'label': 'Top 10', 'value': 10},
                                        {'label': 'Top 15', 'value': 15}
                                    ],
                                    value=10,
                                    style={'width': '120px', 'display': 'inline-block', 'margin-left': '10px'}
                                )
                            ]),
                            className="bg-info text-white"
                        ),
                        dbc.CardBody(
                            dcc.Loading(
                                dcc.Graph(
                                    id='productos_animados',
                                    config={
                                        'displayModeBar': True,
                                        'modeBarButtonsToAdd': ['zoom2d', 'pan2d', 'resetScale2d', 'toImage'],
                                        'scrollZoom': True
                                    },
                                    style={'height': '650px'}
                                ),
                                type="cube"
                            )
                        )
                    ])
                ], md=12)
            ], className="mb-4"),

            
            
            # Almacén de datos para compartir el DataFrame filtrado entre callbacks
            dcc.Store(id='datos-originales', data=df.to_json(date_format='iso', orient='split')),
            dcc.Store(id='datos-filtrados', data=df.to_json(date_format='iso', orient='split')),
        ], fluid=True)
        
    except Exception as e:
        print(f"Error en layout: {str(e)}")
        return dbc.Alert(f"Error al cargar datos: {str(e)}", color="danger")

content = create_content()