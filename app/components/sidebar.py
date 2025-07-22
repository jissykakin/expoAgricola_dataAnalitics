from dash import html
import dash_bootstrap_components as dbc

def create_sidebar():
    return html.Div(
        [
            # Encabezado del sidebar
            html.Div(
                [
                    html.H5("Bienvenido, Jissy", className="text-white mb-4 ps-3 pt-3"),
                    html.Button(
                        html.I(className="fas fa-times text-white"), 
                        id="sidebar-close-btn",
                        className="btn btn-link d-md-none position-absolute",
                        style={"right": "10px", "top": "15px"}
                    )
                ],
                className="position-relative"
            ),
        
            # Contenido del menú
            dbc.Nav(
                [
                    html.Div(
                    [
                        
                            dbc.NavLink(
                                [
                                    html.I(className="fas fa-home me-2 text-white"), 
                                    html.Span("Dashboard", className="text-white")
                                ],
                                href="/",
                                active="exact",
                                className="ps-4 py-2 mb-4"
                            )
                    ]),
                    # Sección Exportaciones
                    html.Div(
                        [
                            html.H6("EXPORTACIONES", className="text-white ps-3 mb-2 small font-weight-bold"),
                            dbc.NavLink(
                                [
                                    html.I(className="fas fa-chart-bar me-2 text-white"), 
                                    html.Span("Análisis Descriptivo", className="text-white")
                                ],
                                href="/exportaciones/descriptivo",
                                active="exact",
                                className="ps-4 py-2"
                            ),
                            dbc.NavLink(
                                [
                                    html.I(className="fas fa-brain me-2 text-white"), 
                                    html.Span("Análisis Predictivo", className="text-white")
                                ],
                                href="/exportaciones/predictivo",
                                active="exact",
                                className="ps-4 py-2"
                            ),
                        ],
                        className="mb-3"
                    ),
                    
                    # Sección Producción
                    html.Div(
                        [
                            html.H6("PRODUCCIÓN", className="text-white ps-3 mb-2 small font-weight-bold"),
                            dbc.NavLink(
                                [
                                    html.I(className="fas fa-chart-pie me-2 text-white"), 
                                    html.Span("Análisis Descriptivo", className="text-white")
                                ],
                                href="/produccion/descriptivo",
                                active="exact",
                                className="ps-4 py-2"
                            ),
                            dbc.NavLink(
                                [
                                    html.I(className="fas fa-robot me-2 text-white"), 
                                    html.Span("Análisis Predictivo", className="text-white")
                                ],
                                href="/produccion/predictivo",
                                active="exact",
                                className="ps-4 py-2"
                            ),
                        ],
                        className="mb-3"
                    ),
                    
                    # Sección Integración
                    html.Div(
                        [
                            html.H6("INTEGRACIÓN", className="text-white ps-3 mb-2 small font-weight-bold"),
                            dbc.NavLink(
                                [
                                    html.I(className="fas fa-project-diagram me-2 text-white"), 
                                    html.Span("Análisis Descriptivo", className="text-white")
                                ],
                                href="/integracion/descriptivo",
                                active="exact",
                                className="ps-4 py-2"
                            ),
                            dbc.NavLink(
                                [
                                    html.I(className="fas fa-network-wired me-2 text-white"), 
                                    html.Span("Análisis Predictivo", className="text-white")
                                ],
                                href="/integracion/predictivo",
                                active="exact",
                                className="ps-4 py-2"
                            ),
                        ]
                    ),
                    
                    # Pie del sidebar
                    html.Div(
                        [
                            html.Hr(className="my-3 bg-light"),
                            dbc.NavLink(
                                [
                                    html.I(className="fas fa-cog me-2 text-white"), 
                                    html.Span("Configuración", className="text-white")
                                ],
                                href="/configuracion",
                                active="exact",
                                className="ps-3 py-2"
                            ),
                        ],
                        className="mt-auto"
                    )
                ],
                vertical=True,
                pills=True,
                className="px-2"
            )
        ],
        id="sidebar",
        style={
            "position": "fixed",
            "top": "0",
            "left": "0",
            "width": "250px",
            "height": "100vh",
            "backgroundColor": "#212529",
            "color": "white",
            "overflowY": "auto",
            "zIndex": "1000",
            "transition": "all 0.3s",
            "boxShadow": "2px 0 10px rgba(0, 0, 0, 0.5)",
            "paddingTop": "70px"
        },
        className="d-flex flex-column"
    )