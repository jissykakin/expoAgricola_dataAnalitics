from dash import html
import dash_bootstrap_components as dbc
from .navbar import create_navbar
from .sidebar import create_sidebar

def create_app_layout(content):
    """Estructura base con navbar y sidebar fijos"""
    return html.Div(
        [
            create_navbar(),  # Navbar siempre visible
            html.Div(
                [
                    create_sidebar(),  # Sidebar siempre visible
                    html.Div(
                        content,
                        id="page-content",
                        style={
                            "margin-left": "18rem",
                            "margin-top": "56px",
                            "padding": "2rem 1rem",
                        },
                    ),
                ],
                style={"display": "flex"}
            )
        ]
    )