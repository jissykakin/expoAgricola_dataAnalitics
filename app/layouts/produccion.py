from dash import html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

def create_layout():
    return dbc.Container([
        html.H1("Análisis de Producción Agrícola", className="mb-4 text-4xl font-bold"),
        
        # Filtros
        dbc.Card([
            dbc.CardHeader("Filtros", className="bg-primary text-white"),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Label("Año"),
                        dcc.Dropdown(
                            id='filtro-ano',
                            multi=True,
                            placeholder="Seleccione año(s)"
                        )
                    ], md=4),
                    
                    dbc.Col([
                        html.Label("Departamento"),
                        dcc.Dropdown(
                            id='filtro-departamento',
                            multi=True,
                            placeholder="Seleccione departamento(s)"
                        )
                    ], md=4),
                    
                    dbc.Col([
                        html.Label("Cultivo"),
                        dcc.Dropdown(
                            id='filtro-cultivo',
                            multi=True,
                            placeholder="Seleccione cultivo(s)"
                        )
                    ], md=4),
                ]),
                
                dbc.Row([
                    dbc.Col([
                        html.Label("Grupo de Cultivo"),
                        dcc.Dropdown(
                            id='filtro-grupo-cultivo',
                            multi=True,
                            placeholder="Seleccione grupo(s)"
                        )
                    ], md=6),
                    
                    dbc.Col([
                        html.Label("Tipo de Periodo"),
                        dcc.Dropdown(
                            id='filtro-tipo-periodo',
                            options=[
                                {'label': 'Semestral', 'value': 'Semestral'},
                                {'label': 'Anual', 'value': 'Anual'},
                                {'label': 'Todos', 'value': 'Todos'}
                            ],
                            value='Todos'
                        )
                    ], md=6),
                ], className="mt-3"),
                
                dbc.Row([
                    dbc.Col(
                        dbc.Button("Aplicar Filtros", id="boton-filtrar", color="primary", className="w-100"),
                        md=12
                    )
                ], className="mt-3")
            ])
        ], className="mb-4"),
        
        # Gráficos - NUEVO GRÁFICO COMPARATIVO AL INICIO
        dbc.Row([
            dbc.Col(
                dbc.Card([
                    dbc.CardHeader(
                        "Comparativa de Área, Producción y Rendimiento",
                        className="bg-success text-white"
                    ),
                    dbc.CardBody(
                        dcc.Graph(
                            id='grafico-comparativo',
                            config={'displayModeBar': True},
                            style={'height': '550px'}
                        )
                    )
                ]),
                md=12
            )
        ], className="mb-4"),
        
        # Gráficos existentes      
        
        dbc.Row([
            dbc.Col(dcc.Graph(id='grafico-departamentos'), md=6),
            dbc.Col(dcc.Graph(id='grafico-distribucion'), md=6),
        ], className="mb-4"),
        
        dbc.Row([
            dbc.Col(dcc.Graph(id='grafico-tendencia'), md=12),
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(id='grafico-evolucion'), md=12),
        ], className="mb-4"),
        
        # Almacenes de datos
        dcc.Store(id='data-originales'),
        dcc.Store(id='data-filtrados')
    ], fluid=True)

content = create_layout()

# from dash import html, dcc
# import dash_bootstrap_components as dbc
# from dash.dependencies import Input, Output

# def create_layout():
#     return dbc.Container([
#         html.H1("Análisis de Producción Agrícola", className="mb-4 text-center"),
        
#         # Filtros
#         dbc.Card([
#             dbc.CardHeader("Filtros", className="bg-primary text-white"),
#             dbc.CardBody([
#                 dbc.Row([
#                     dbc.Col([
#                         html.Label("Año"),
#                         dcc.Dropdown(
#                             id='filtro-ano',
#                             multi=True,
#                             placeholder="Seleccione año(s)"
#                         )
#                     ], md=4),
                    
#                     dbc.Col([
#                         html.Label("Departamento"),
#                         dcc.Dropdown(
#                             id='filtro-departamento',
#                             multi=True,
#                             placeholder="Seleccione departamento(s)"
#                         )
#                     ], md=4),
                    
#                     dbc.Col([
#                         html.Label("Cultivo"),
#                         dcc.Dropdown(
#                             id='filtro-cultivo',
#                             multi=True,
#                             placeholder="Seleccione cultivo(s)"
#                         )
#                     ], md=4),
#                 ]),
                
#                 dbc.Row([
#                     dbc.Col([
#                         html.Label("Grupo de Cultivo"),
#                         dcc.Dropdown(
#                             id='filtro-grupo-cultivo',
#                             multi=True,
#                             placeholder="Seleccione grupo(s)"
#                         )
#                     ], md=6),
                    
#                     dbc.Col([
#                         html.Label("Tipo de Periodo"),
#                         dcc.Dropdown(
#                             id='filtro-tipo-periodo',
#                             options=[
#                                 {'label': 'Semestral', 'value': 'Semestral'},
#                                 {'label': 'Anual', 'value': 'Anual'},
#                                 {'label': 'Todos', 'value': 'Todos'}
#                             ],
#                             value='Todos'
#                         )
#                     ], md=6),
#                 ], className="mt-3"),
                
#                 dbc.Row([
#                     dbc.Col(
#                         dbc.Button("Aplicar Filtros", id="boton-filtrar", color="primary", className="w-100"),
#                         md=12
#                     )
#                 ], className="mt-3")
#             ])
#         ], className="mb-4"),
        
#         # Gráficos
#         dbc.Row([
#             dbc.Col(dcc.Graph(id='grafico-evolucion'), md=12),
#         ], className="mb-4"),
        
#         dbc.Row([
#             dbc.Col(dcc.Graph(id='grafico-departamentos'), md=6),
#             dbc.Col(dcc.Graph(id='grafico-distribucion'), md=6),
#         ], className="mb-4"),
        
#         dbc.Row([
#             dbc.Col(dcc.Graph(id='grafico-tendencia'), md=12),
#         ]),
        
#         # Almacenes de datos
#         dcc.Store(id='data-originales'),
#         dcc.Store(id='data-filtrados')
#     ], fluid=True)

# content = create_layout()