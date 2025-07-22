import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re
from datetime import datetime
from typing import Optional

class AnalisisProduccion:
    def __init__(self, df):
        self.df = self._preparar_datos(df.copy())
    
    def _preparar_datos(self, df):
        """Limpieza y preparación de datos con procesamiento de periodo"""
        # Convertir columnas a numéricas
        numeric_cols = ['area_sembrada_(ha)', 'area_cosechada_(ha)', 'produccion_(t)', 'rendimiento_(t/ha)']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Procesar el campo periodo
        df['semestre'] = df['periodo'].apply(self._extraer_semestre)
        df['tipo_periodo'] = df['periodo'].apply(self._determinar_tipo_periodo)
        
        return df.dropna(subset=numeric_cols)
    
    def _extraer_semestre(self, periodo):
        """Extrae semestre de formato 2022A, 2022B o 2022"""
        if pd.isna(periodo):
            return None
        
        periodo = str(periodo).strip().upper()
        
        if periodo.endswith('A'):
            return 1
        elif periodo.endswith('B'):
            return 2
        else:
            return None  # Para años completos
    
    def _determinar_tipo_periodo(self, periodo):
        """Clasifica el periodo en Semestral/Anual"""
        if pd.isna(periodo):
            return None
        
        periodo = str(periodo).strip().upper()
        
        if 'A' in periodo or 'B' in periodo:
            return 'Semestral'
        else:
            return 'Anual'
    
    # def grafico_evolucion_rendimiento(self, cultivos_seleccionados=None, departamento=None):
    #     """Evolución del rendimiento por cultivo y periodo"""
    #     df_filtrado = self.df.copy()
        
    #     if cultivos_seleccionados:
    #         df_filtrado = df_filtrado[df_filtrado['cultivo'].isin(cultivos_seleccionados)]
        
    #     if departamento:
    #         df_filtrado = df_filtrado[df_filtrado['departamento'] == departamento]
        
    #     evolucion = df_filtrado.groupby(['ano', 'periodo', 'cultivo'])['rendimiento_(t/ha)'].mean().reset_index()
        
    #     fig = px.line(
    #         evolucion,
    #         x='ano',
    #         y='rendimiento_(t/ha)',
    #         color='cultivo',
    #         line_dash='periodo',
    #         title='Evolución del Rendimiento por Cultivo y Periodo',
    #         markers=True,
    #         labels={
    #             'ano': 'Año',
    #             'rendimiento_(t/ha)': 'Rendimiento (t/ha)',
    #             'periodo': 'Periodo',
    #             'cultivo': 'Cultivo'
    #         }
    #     )
    #     fig.update_layout(hovermode='x unified')
    #     return fig

    def grafico_comparativo_rendimiento(self, top_n=10, departamento=None, grupo_cultivo=None):
        """
        Versión mejorada con más interactividad
        """
        df_filtrado = self.df.copy()
        
        # Aplicar filtros
        if departamento:
            df_filtrado = df_filtrado[df_filtrado['departamento'] == departamento]
        
        if grupo_cultivo:
            df_filtrado = df_filtrado[df_filtrado['grupo_cultivo'] == grupo_cultivo]
        
        # Agrupar y procesar datos
        datos_agrupados = df_filtrado.groupby('cultivo').agg({
            'area_sembrada_(ha)': 'sum',
            'produccion_(t)': 'sum',
            'rendimiento_(t/ha)': 'mean'
        }).reset_index()
        
        # Seleccionar top N y ordenar
        datos_agrupados = datos_agrupados.nlargest(top_n, 'area_sembrada_(ha)')
        datos_agrupados = datos_agrupados.sort_values('area_sembrada_(ha)', ascending=False)
        
        # Crear figura
        fig = go.Figure()
        
        # Barras para área sembrada (eje izquierdo)
        fig.add_trace(go.Bar(
            x=datos_agrupados['cultivo'],
            y=datos_agrupados['area_sembrada_(ha)'],
            name='Área Sembrada (ha)',
            marker_color='#1f77b4',
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Área Sembrada: %{y:,.0f} ha<br>"
                "<extra></extra>"
            )
        ))
        
        # Barras para producción (eje izquierdo)
        fig.add_trace(go.Bar(
            x=datos_agrupados['cultivo'],
            y=datos_agrupados['produccion_(t)'],
            name='Producción (t)',
            marker_color='#ff7f0e',
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Producción: %{y:,.0f} t<br>"
                "<extra></extra>"
            )
        ))
        
        # Línea para rendimiento (eje derecho)
        fig.add_trace(go.Scatter(
            x=datos_agrupados['cultivo'],
            y=datos_agrupados['rendimiento_(t/ha)'],
            name='Rendimiento (t/ha)',
            mode='lines+markers',
            line=dict(color='#2ca02c', width=3),
            yaxis='y2',
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Rendimiento: %{y:.2f} t/ha<br>"
                "<extra></extra>"
            )
        ))
        
        # Personalizar diseño
        title = f'Comparación de Top {top_n} Cultivos'
        if departamento:
            title += f' en {departamento}'
        if grupo_cultivo:
            title += f' ({grupo_cultivo})'
        
        fig.update_layout(
            title=f'<b>{title}</b>',
            barmode='group',
            plot_bgcolor='white',
            xaxis=dict(
                title='<b>Cultivo</b>',
                type='category',
                tickangle=45
            ),
            yaxis=dict(
                title='<b>Área (ha) / Producción (t)</b>',
                showgrid=True,
                gridcolor='lightgray'
            ),
            yaxis2=dict(
                title='<b>Rendimiento (t/ha)</b>',
                overlaying='y',
                side='right',
                showgrid=False,
                range=[0, datos_agrupados['rendimiento_(t/ha)'].max() * 1.2]
            ),
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor='rgba(255,255,255,0.5)'
            ),
            margin=dict(l=100, r=100, t=100, b=150),
            dragmode='pan'
        )
        
        # Añadir anotación con resumen
        fig.add_annotation(
            x=0.5,
            y=-0.3,
            xref="paper",
            yref="paper",
            text="Área Sembrada (ha) | Producción (t) | Rendimiento (t/ha)",
            showarrow=False,
            font=dict(size=12)
        )
        
        return fig

    def grafico_evolucion_rendimiento(self, cultivos_seleccionados=None, departamento=None):
        """Evolución del rendimiento por cultivo y periodo (versión mejorada)"""
        df_filtrado = self.df.copy()
        
        # Aplicar filtros
        if cultivos_seleccionados:
            df_filtrado = df_filtrado[df_filtrado['cultivo'].isin(cultivos_seleccionados)]
        
        if departamento:
            df_filtrado = df_filtrado[df_filtrado['departamento'] == departamento]
        
        # Procesamiento de datos
        evolucion = df_filtrado.groupby(['ano', 'periodo', 'cultivo', 'semestre'])['rendimiento_(t/ha)'].mean().reset_index()
        
        # Crear columna combinada de año-semestre para el eje X
        evolucion['periodo_visual'] = evolucion['ano'].astype(str) + \
                                    evolucion['semestre'].apply(lambda x: f"-S{int(x)}" if pd.notnull(x) else "")
        
        # Ordenar por año y semestre
        evolucion = evolucion.sort_values(['ano', 'semestre'])
        
        # Crear gráfico interactivo
        fig = px.line(
            evolucion,
            x='periodo_visual',
            y='rendimiento_(t/ha)',
            color='cultivo',
            line_dash='periodo',
            title='<b>Evolución Temporal del Rendimiento Agrícola</b>',
            markers=True,
            labels={
                'periodo_visual': 'Periodo (Año-Semestre)',
                'rendimiento_(t/ha)': 'Rendimiento (toneladas/hectárea)',
                'cultivo': 'Cultivo',
                'periodo': 'Tipo de Periodo'
            },
            hover_data={
                'ano': True,
                'semestre': True,
                'periodo': True,
                'periodo_visual': False
            }
        )
        
        # Mejorar diseño y legibilidad
        fig.update_layout(
            hovermode='x unified',
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis_title='<b>Periodo</b>',
            yaxis_title='<b>Rendimiento (t/ha)</b>',
            legend_title='<b>Cultivo</b>',
            font=dict(
                family="Arial",
                size=12,
                color="#333333"
            ),
            xaxis=dict(
                showline=True,
                showgrid=True,
                showticklabels=True,
                linecolor='rgb(204, 204, 204)',
                linewidth=2,
                ticks='outside',
                tickfont=dict(
                    family='Arial',
                    size=12,
                    color='rgb(82, 82, 82)',
                ),
            ),
            yaxis=dict(
                showline=True,
                showgrid=True,
                showticklabels=True,
                linecolor='rgb(204, 204, 204)',
                linewidth=2,
                gridcolor='rgb(230, 230, 230)',
            ),
            margin=dict(autoexpand=True, l=100, r=100, t=110),
            showlegend=True
        )
        
        # Mejorar tooltips
        fig.update_traces(
            hovertemplate=(
                "<b>%{fullData.name}</b><br>"
                "Año: %{customdata[0]}<br>"
                "Semestre: %{customdata[1]}<br>"
                "Periodo: %{customdata[2]}<br>"
                "Rendimiento: %{y:.2f} t/ha"
                "<extra></extra>"
            )
        )
        
        # Añadir anotación con los filtros aplicados
        if departamento or cultivos_seleccionados:
            filtros = []
            if departamento:
                filtros.append(f"Departamento: {departamento}")
            if cultivos_seleccionados and len(cultivos_seleccionados) < 5:
                filtros.append(f"Cultivos: {', '.join(cultivos_seleccionados)}")
            
            if filtros:
                fig.add_annotation(
                    x=0.5,
                    y=1.15,
                    xref="paper",
                    yref="paper",
                    text=" | ".join(filtros),
                    showarrow=False,
                    font=dict(
                        size=12,
                        color="#666666"
                    )
                )
        
        return fig
        
    def grafico_comparacion_departamentos(self, cultivo=None, ano=None):
        """Comparación de rendimiento por departamento"""
        df_filtrado = self.df.copy()
        
        if cultivo:
            df_filtrado = df_filtrado[df_filtrado['cultivo'] == cultivo]
        
        if ano:
            df_filtrado = df_filtrado[df_filtrado['ano'] == ano]
        
        deptos = df_filtrado.groupby('departamento').agg({
            'rendimiento_(t/ha)': 'mean',
            'produccion_(t)': 'sum'
        }).reset_index()
        
        fig = px.treemap(
            deptos,
            path=['departamento'],
            values='produccion_(t)',
            color='rendimiento_(t/ha)',
            title='Producción y Rendimiento por Departamento',
            color_continuous_scale='Blues',
            labels={
                'departamento': 'Departamento',
                'produccion_(t)': 'Producción Total (t)',
                'rendimiento_(t/ha)': 'Rendimiento Promedio (t/ha)'
            }
        )
        return fig
    
    def grafico_distribucion_cultivos(self, grupo_cultivo=None):
        """Distribución de cultivos por área sembrada"""
        df_filtrado = self.df.copy()
        
        if grupo_cultivo:
            df_filtrado = df_filtrado[df_filtrado['grupo_cultivo'] == grupo_cultivo]
        
        cultivos = df_filtrado.groupby('cultivo').agg({
            'area_sembrada_(ha)': 'sum',
            'rendimiento_(t/ha)': 'mean'
        }).reset_index()
        
        fig = px.sunburst(
            cultivos,
            path=['cultivo'],
            values='area_sembrada_(ha)',
            color='rendimiento_(t/ha)',
            title='Distribución de Cultivos por Área Sembrada',
            color_continuous_scale='RdBu',
            labels={
                'area_sembrada_(ha)': 'Área Sembrada (ha)',
                'rendimiento_(t/ha)': 'Rendimiento (t/ha)'
            }
        )
        return fig
    
    def grafico_tendencia_anual(self, metricas=None):
        """Tendencia anual de métricas clave"""
        if metricas is None:
            metricas = ['area_sembrada_(ha)', 'produccion_(t)', 'rendimiento_(t/ha)']
        
        tendencia = self.df.groupby('ano')[metricas].mean().reset_index()
        
        fig = go.Figure()
        
        for metrica in metricas:
            fig.add_trace(go.Scatter(
                x=tendencia['ano'],
                y=tendencia[metrica],
                name=metrica,
                mode='lines+markers'
            ))
        
        fig.update_layout(
            title='Tendencia Anual de Métricas Agrícolas',
            xaxis_title='Año',
            yaxis_title='Valor',
            hovermode='x unified'
        )
        return fig