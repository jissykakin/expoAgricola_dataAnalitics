from dash import html, dcc
import dash_bootstrap_components as dbc

def create_layout():
    return dbc.Container([
        html.H1("Dashboard Agrícola - Resumen General", className="mb-4 text-center"),
        
        # Sección de métricas
        dbc.Row([
            dbc.Col(
                dbc.Card([
                    dbc.CardHeader("Producción Agrícola", className="bg-primary text-white "),
                    dbc.CardBody([
                        html.Div(id='metricas-produccion', className="metricas-container text-xs text-bold")
                    ])
                ]),
                md=6
            ),
            dbc.Col(
                dbc.Card([
                    dbc.CardHeader("Exportaciones", className="bg-success text-white"),
                    dbc.CardBody([
                        html.Div(id='metricas-exportaciones', className="metricas-container text-xs text-bold")
                    ])
                ]),
                md=6
            )
        ], className="mb-4"),
        
        # Gráficos principales
        dbc.Row([
            dbc.Col(
                dbc.Card([
                    dbc.CardHeader("Resumen Producción Agrícola", className="bg-info text-white"),
                    dbc.CardBody([
                        dcc.Graph(id='grafico-resumen-produccion')
                    ])
                ]),
                md=6
            ),
            dbc.Col(
                dbc.Card([
                    dbc.CardHeader("Resumen Exportaciones", className="bg-warning text-dark"),
                    dbc.CardBody([
                        dcc.Graph(id='grafico-resumen-exportaciones')
                    ])
                ]),
                md=6
            )
        ], className="mb-4"),
        
        # Gráficos top
        dbc.Row([
            dbc.Col(
                dbc.Card([
                    dbc.CardHeader("Top Cultivos", className="bg-danger text-white"),
                    dbc.CardBody([
                        dcc.Graph(id='grafico-top-cultivos')
                    ])
                ]),
                md=6
            ),
            dbc.Col(
                dbc.Card([
                    dbc.CardHeader("Top Exportaciones", className="bg-secondary text-white"),
                    dbc.CardBody([
                        dcc.Graph(id='grafico-top-exportaciones')
                    ])
                ]),
                md=6
            ), 
        ]),
                
        # Almacenes de datos
        dcc.Store(id='trigger-load', data=True),
        dcc.Store(id='store-datos-produccion'),
        dcc.Store(id='store-datos-exportaciones'),
        
    ], fluid=True, className="dashboard-container")

content = create_layout()