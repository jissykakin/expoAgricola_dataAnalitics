# app/app.py
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
from components.navbar import create_navbar
from components.sidebar import create_sidebar
from layouts.layout_dashboard import content as main_content
from layouts.exportaciones import content as exportaciones_descriptivo_content
from layouts.produccion import content as produccion_descriptivo_content
from layouts.exportacion_predictive import content as exportaciones_predictivo_content
from layouts.produccion_predictive import content as produccion_predictivo_content
from layouts.intregacion_descriptivo import content as integracion_content
from callbacks.exportaciones import register_exportaciones_callbacks
from callbacks.predictive_exportaciones import register_predictive_callbacks
from callbacks.callbacks_produccion import register_callbacks
from callbacks.callbacks_dashboard import register_callbacks_dashboard
from callbacks.advanced_predictive import register_advanced_predictive_callbacks


app = Dash(__name__, external_stylesheets=[
    dbc.themes.BOOTSTRAP,
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css"
])
app.config.suppress_callback_exceptions = True


app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    create_navbar(),
    create_sidebar(),
    html.Div(id='page-content', style={
        'marginLeft': '18rem',
        'marginTop': '10px',
        'padding': '1rem'
    })
])

register_callbacks_dashboard(app)
register_exportaciones_callbacks(app)
register_predictive_callbacks(app)
register_advanced_predictive_callbacks(app)
register_callbacks(app)

@app.callback(
    Output('page-content', 'children'),
    Input('url', 'pathname'),
    prevent_initial_call=True
)
def update_content(pathname):
    try:
        if pathname is None:
            return exportaciones_descriptivo_content
        
        # Definir el mapeo de rutas a contenidos
        route_mapping = {
            '/': main_content,
            '/exportaciones/descriptivo': exportaciones_descriptivo_content,
            '/exportaciones/predictivo': exportaciones_predictivo_content,
            '/produccion/descriptivo': produccion_descriptivo_content,
            '/produccion/predictivo': produccion_predictivo_content,
            '/integracion/descriptivo': integracion_content
        }
        
        # Verificar si la ruta existe en el mapeo
        if pathname in route_mapping:
            return route_mapping[pathname]
        
        # Si la ruta no existe, mostrar contenido por defecto con mensaje
        return html.Div([
            html.H1("404 - Página no encontrada", className="text-danger"),
            html.P(f"No se encontró la ruta: {pathname}"),
            dbc.Button("Volver al inicio", href="/", color="primary")
        ], className="text-center p-5")
        
    except Exception as e:
        # Manejo de errores detallado
        error_message = html.Div([
            html.H1("Error al cargar la página", className="text-danger"),
            html.P("Ocurrió un error al procesar su solicitud."),
            html.P(str(e), className="text-muted small"),
            dbc.Button("Volver al inicio", href="/", color="primary")
        ], className="text-center p-5")
        
        return error_message

if __name__ == '__main__':
    app.run(debug=True)