import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

class DashboardAnalitico:
    def __init__(self, df_produccion, df_exportaciones):
        self.df_prod = self._preparar_datos_produccion(df_produccion.copy())
        self.df_exp = self._preparar_datos_exportaciones(df_exportaciones.copy())
    
    def _preparar_datos_produccion(self, df):
        """Preparación datos producción agrícola"""
        numeric_cols = ['area_sembrada_(ha)', 'area_cosechada_(ha)', 'produccion_(t)', 'rendimiento_(t/ha)']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df.dropna(subset=numeric_cols)
    
    def _preparar_datos_exportaciones(self, df):
        """Preparación datos exportaciones"""
        numeric_cols = ['valor_usd', 'volumen_ton']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df.dropna(subset=numeric_cols)
    
    def generar_metricas(self):
        """Genera métricas resumen para ambas tablas"""
        metricas = {
            'produccion': {
                'total_registros': len(self.df_prod),
                'total_cultivos': self.df_prod['cultivo'].nunique(),
                'total_departamentos': self.df_prod['departamento'].nunique(),
                'area_total_sembrada': self.df_prod['area_sembrada_(ha)'].sum(),
                'produccion_total': self.df_prod['produccion_(t)'].sum(),
                'ultimo_ano': self.df_prod['ano'].max()
            },
            'exportaciones': {
                'total_registros': len(self.df_exp),
                'total_productos': self.df_exp['producto'].nunique(),               
                'valor_total_exportado': self.df_exp['valor_usd'].sum(),
                'volumen_total': self.df_exp['volumen_ton'].sum(),
                'ultimo_ano': self.df_exp['año'].max()
            }
        }
        return metricas
    
    def grafico_resumen_produccion(self):
        """Gráfico de resumen producción agrícola"""
        datos = self.df_prod.groupby('ano').agg({
            'area_sembrada_(ha)': 'sum',
            'produccion_(t)': 'sum'
        }).reset_index()
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=datos['ano'],
            y=datos['area_sembrada_(ha)'],
            name='Área Sembrada (ha)',
            marker_color='#1f77b4'
        ))
        
        fig.add_trace(go.Bar(
            x=datos['ano'],
            y=datos['produccion_(t)'],
            name='Producción (t)',
            marker_color='#ff7f0e',
            yaxis='y2'
        ))
        
        fig.update_layout(
            title='<b>Evolución Anual de Producción Agrícola</b>',
            barmode='group',
            plot_bgcolor='white',
            xaxis_title='Año',
            yaxis=dict(
                title='Área Sembrada (ha)',
                side='left'
            ),
            yaxis2=dict(
                title='Producción (t)',
                side='right',
                overlaying='y'
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    def grafico_resumen_exportaciones(self):
        """Gráfico de resumen exportaciones"""
        datos = self.df_exp.groupby('año').agg({
            'valor_usd': 'sum',
            'volumen_ton': 'sum'
        }).reset_index()
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=datos['año'],
            y=datos['valor_usd'],
            name='Valor (USD)',
            marker_color='#2ca02c'
        ))
        
        fig.add_trace(go.Bar(
            x=datos['año'],
            y=datos['volumen_ton'],
            name='Volumen (ton)',
            marker_color='#d62728',
            yaxis='y2'
        ))
        
        fig.update_layout(
            title='<b>Evolución Anual de Exportaciones</b>',
            barmode='group',
            plot_bgcolor='white',
            xaxis_title='Año',
            yaxis=dict(
                title='Valor (USD)',
                side='left'
            ),
            yaxis2=dict(
                title='Volumen (ton)',
                side='right',
                overlaying='y'
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    def grafico_top_cultivos(self, top_n=5):
        """Top cultivos por área sembrada"""
        datos = self.df_prod.groupby('cultivo').agg({
            'area_sembrada_(ha)': 'sum'
        }).nlargest(top_n, 'area_sembrada_(ha)').reset_index()
        
        fig = px.bar(
            datos,
            x='cultivo',
            y='area_sembrada_(ha)',
            title=f'<b>Top {top_n} Cultivos por Área Sembrada</b>',
            color='cultivo',
            labels={'area_sembrada_(ha)': 'Área Sembrada (ha)', 'cultivo': 'Cultivo'}
        )
        
        fig.update_layout(showlegend=False)
        return fig
    
    def grafico_top_exportaciones(self, top_n=5):
        """Top productos de exportación"""
        datos = self.df_exp.groupby('producto').agg({
            'valor_usd': 'sum'
        }).nlargest(top_n, 'valor_usd').reset_index()
        
        fig = px.bar(
            datos,
            x='producto',
            y='valor_usd',
            title=f'<b>Top {top_n} Productos de Exportación</b>',
            color='producto',
            labels={'valor_usd': 'Valor (USD)', 'producto': 'Producto'}
        )
        
        fig.update_layout(showlegend=False)
        return fig