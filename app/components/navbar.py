# components/navbar.py
import dash_bootstrap_components as dbc
from dash import html


def create_navbar():
    navbar = dbc.Navbar(
        [
            # Logo e Identidad
            html.Div([
                html.Img(src="/assets/logos.png", height="12px", style={"marginRight": "10px"}),
                html.Span("AgriDash", className="navbar-brand mb-0 h1", style={"fontWeight": "bold", "fontSize": "24px"})
            ], style={"display": "flex", "alignItems": "center"}),

            # Iconos de usuario, notificación, salir
            dbc.NavbarToggler(id="navbar-toggler"),
            dbc.Collapse(
                dbc.Nav([                    
                    dbc.NavItem(dbc.NavLink(html.I(className="fas fa-sliders-h me-2 text-white"), href="#", style={"fontSize": "24px"})),
                    dbc.NavItem(dbc.NavLink(html.I(className="fa-regular fa-circle-user me-2 text-white"), href="#", style={"fontSize": "24px"})),
                    dbc.NavItem(dbc.NavLink(html.I(className="fas fa-sign-out-alt me-2 text-white"), href="#", style={"fontSize": "24px"})),
                    # dbc.Button("Salir", color="danger", outline=True, size="sm", className="ms-3")
                ], className="ms-auto", navbar=True),
                id="navbar-collapse",
                navbar=True
            )
        ],
        color="dark",
        dark=True,
        fixed="top",        
        className="shadow-sm px-2"
    )
    return navbar