from dash import Dash
from app.layout import layout

app = Dash(__name__, assets_folder='app/assets')
app.title = "AgroAnalytics"
app.layout = layout

if __name__ == '__main__':
    app.run_server(debug=True)