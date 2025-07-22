from dash import html, dcc
import dash_bootstrap_components as dbc
from components.navbar import create_navbar
from components.sidebar import create_sidebar

layout_default = html.Div([
    dcc.Location(id='url', refresh=False),

    # Navbar superior
    create_navbar(),

    # Layout principal con sidebar y contenido
    html.Div([
        # Sidebar izquierdo
        html.Div(
            create_sidebar(),
            id="sidebar",
            className="bg-light d-none d-md-block",
            style={
                "width": "260px",
                "position": "fixed",
                "top": "56px",  # Altura del navbar
                "bottom": 0,
                "left": 0,
                "overflowY": "auto",
                "padding": "1rem"
            }
        ),

        # Contenido principal
        html.Div(
            id="page-content",
            style={
                "marginLeft": "0",
                "marginTop": "0px",  # Altura del navbar
                "padding": "0rem"
            },
            className="ml-md-0"
        )
    ], style={"display": "flex", "flexDirection": "row"})
])