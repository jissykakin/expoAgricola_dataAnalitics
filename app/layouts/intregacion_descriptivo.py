from dash import dcc, html
import dash_bootstrap_components as dbc

def create_content():
    return dbc.Container([
            html.H2("Análisis Descriptivo de Exportaciones", className="mb-4 text-4xl font-bold"),
    ])

content = create_content()  