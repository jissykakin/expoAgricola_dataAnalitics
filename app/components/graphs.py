import plotly.express as px
import plotly.graph_objects as go
from typing import Dict

def create_time_series(df, x_col, y_col, title):
    fig = px.line(df, x=x_col, y=y_col, title=title)
    fig.update_layout(template="plotly_white")
    return fig

def apply_style(fig: go.Figure, config: Dict) -> go.Figure:
    """Aplica estilos consistentes a los gráficos"""
    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        hovermode='x unified'
    )
    return fig