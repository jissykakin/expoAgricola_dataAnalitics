import pandas as pd
import plotly.express as px
from typing import Tuple, Dict

class AnalisisExportaciones:
    def __init__(self, df):
        # Validación de datos inicial
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Se debe proporcionar un DataFrame de pandas")
        
        self.df = df.copy()
        self._preparar_datos()
    
    def _preparar_datos(self):
        """Prepara y valida los datos"""
        required_columns = ['año', 'mes', 'valor_usd', 'tipo_producto', 'departamento']
        missing = [col for col in required_columns if col not in self.df.columns]
        if missing:
            raise ValueError(f"Faltan columnas requeridas: {missing}")

        # Mapeo de meses (asegurar formato consistente)
        meses_espanol = {
            'enero': 1, 'febrero': 2, 'marzo': 3, 'abril': 4,
            'mayo': 5, 'junio': 6, 'julio': 7, 'agosto': 8,
            'septiembre': 9, 'octubre': 10, 'noviembre': 11, 'diciembre': 12
        }
        
        # Normalización de meses
        self.df['mes'] = self.df['mes'].str.strip().str.lower()
        self.df['mes_numero'] = self.df['mes'].map(meses_espanol)
        
        if self.df['mes_numero'].isna().any():
            invalid = self.df[self.df['mes_numero'].isna()]['mes'].unique()
            raise ValueError(f"Meses no reconocidos: {invalid}")
        
        # Crear fecha
        self.df['fecha'] = pd.to_datetime(
            self.df['año'].astype(str) + '-' + 
            self.df['mes_numero'].astype(str) + '-01'
        )
    
    def _apply_style(self, fig):
        """Aplica estilos consistentes"""
        fig.update_layout(
            margin=dict(l=50, r=50, t=80, b=50),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Arial", size=14, color="gray"),
            hoverlabel=dict(bgcolor="#2a3f5f", font_size=14)
        )
        return fig
    def _apply_style2(self, fig):
        """Aplica estilos con énfasis en visibilidad de labels"""
        fig.update_layout(
            # Configuración base
            margin=dict(l=50, r=50, t=80, b=80),  # Más margen para labels
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0.5)',
            
            # Estilos de texto
            font=dict(
                family="Arial",
                size=14,
                color="white"  # Color base para textos
            ),
            
            # Ejes
            xaxis=dict(
                title_font=dict(size=16, color='white'),
                tickfont=dict(size=14, color='white'),
                gridcolor='rgba(128, 128, 128, 0.3)',
                linecolor='rgba(128, 128, 128, 0.5)',
                zerolinecolor='rgba(128, 128, 128, 0.5)'
            ),
            yaxis=dict(
                title_font=dict(size=16, color='white'),
                tickfont=dict(size=14, color='white'),
                gridcolor='rgba(128, 128, 128, 0.3)',
                linecolor='rgba(128, 128, 128, 0.5)',
                zerolinecolor='rgba(128, 128, 128, 0.5)'
            ),
            
            # Leyenda
            legend=dict(
                font=dict(size=14, color='white'),
                bgcolor='rgba(0,0,0,0.5)',
                bordercolor='rgba(255,255,255,0.3)',
                borderwidth=1
            ),
            
            # Título
            title=dict(
                x=0.5,  # Centrado
                xanchor='center',
                font=dict(size=20, color='white')
            )
        )
        
        # Asegurar contraste en textos de gráficos
        if hasattr(fig.data[0], 'textfont'):
            fig.update_traces(
                textfont=dict(color='white', size=14),
                textposition='auto'  # Posición automática para mejor visibilidad
            )
        
        return fig
    def exportaciones_anuales(self):
        try:
            if self.df.empty:
                return self._create_empty_figure("No hay datos disponibles")
            
            df_anual = self.df.groupby('año', as_index=False)['valor_usd'].sum()
            
            fig = px.bar(
                df_anual,
                x='año',
                y='valor_usd',
                text='valor_usd',  # Asegurar que muestra los valores
                title='<b>Exportaciones por Año (USD FOB)</b>'
            )
            
            fig.update_traces(
                texttemplate='%{text:$,.0f}',  # Formato monetario
                textposition='outside',       # Fuera de las barras
                marker_color='#636EFA',
                textfont=dict(size=14, color='white')  # Estilo específico para labels
            )
        
            return self._apply_style(fig)
        except Exception as e:
            print(f"Error al generar gráfico anual: {str(e)}")
            return px.Figure()
        

    def exportaciones_por_tipo(self):
        try:
            if self.df.empty:
                return self._create_empty_figure("No hay datos disponibles")
            
            df_tipo = self.df.groupby('tipo_producto', as_index=False)['valor_usd'].sum()
            
            fig = px.pie(
                df_tipo,
                names='tipo_producto',
                values='valor_usd',
                title='<b>Distribución por Tipo de Producto</b>',
                hole=0.4
            )
            
            fig.update_traces(
                textinfo='percent+label',
                textfont=dict(size=16, color='white'),
                marker=dict(line=dict(color='#444', width=2)),
                hovertemplate='<b>%{label}</b><br>%{percent:.1%}<br>Valor: $%{value:,.0f}'
            )
            
            return self._apply_style(fig)
        except Exception as e:
            print(f"Error al generar gráfico anual: {str(e)}")
            return px.Figure()  

    def evolucion_mensual(self):
        """Evolución mensual de exportaciones"""
        if self.df.empty:
                return self._create_empty_figure("No hay datos disponibles")
        
        df_mensual = self.df.groupby('fecha', as_index=False)['valor_usd'].sum()
        
        fig = px.line(
            df_mensual,
            x='fecha',
            y='valor_usd',
            title='<b>Evolución Mensual de Exportaciones</b>',
            labels={'valor_usd': 'Valor (USD)', 'fecha': 'Fecha'},
            template='plotly_dark'
        )
        fig.update_traces(line_color='#00CC96')
        return self._apply_style(fig)
    
    def exportaciones_por_departamento(self):
        """Exportaciones por departamento"""
        if self.df.empty:
                return self._create_empty_figure("No hay datos disponibles")
        
        df_depto = self.df.groupby('departamento', as_index=False)['valor_usd'].sum() \
                   .nlargest(10, 'valor_usd')
        
        fig = px.bar(
            df_depto,
            x='valor_usd',
            y='departamento',
            orientation='h',
            title='<b>Top 10 Departamentos Exportadores</b>',
            labels={'valor_usd': 'Valor (USD)', 'departamento': ''},
            template='plotly_dark',
            color='valor_usd'
        )
        return self._apply_style(fig)        
              
    def evolucion_productos_exportados(self, top_n=10):
        """Muestra la evolución anual de los productos más exportados"""
        try:
            # Agrupar por año y producto
            df_agrupado = self.df.groupby(['año', 'producto'], as_index=False)['valor_usd'].sum()
            
            # Obtener los top_n productos más exportados en todo el periodo
            top_productos = df_agrupado.groupby('producto')['valor_usd'].sum().nlargest(top_n).index.tolist()
            
            # Filtrar solo los productos top
            df_filtrado = df_agrupado[df_agrupado['producto'].isin(top_productos)]
            
            # Crear gráfico de líneas temporales
            fig = px.line(
                df_filtrado,
                x='año',
                y='valor_usd',
                color='producto',
                title=f'<b>Evolución Anual de los {top_n} Productos Más Exportados</b>',
                labels={'valor_usd': 'Valor Exportado (USD)', 'año': 'Año', 'producto': 'Producto'},
                hover_data={'valor_usd': ':$,.0f'},
                markers=True
            )
            
            # Mejorar estilo
            fig.update_traces(
                hovertemplate='<b>%{fullData.name}</b><br>Año: %{x}<br>Valor: %{y:$,.0f}<extra></extra>',
                line_width=2.5,
                marker_size=8
            )
            
            fig.update_layout(
                hovermode='x unified',
                height=600,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.3,
                    xanchor="right",
                    x=1
                ),
                xaxis=dict(
                    dtick=1  # Mostrar todos los años
                )
            )
            
            return self._apply_style2(fig)
            
        except Exception as e:
            print(f"Error al generar gráfico de evolución: {str(e)}")
            return self._create_empty_figure("Error al cargar datos de productos")
        

    def productos_animados(self, top_n=10):
        """Gráfico animado de evolución de productos más exportados"""
        try:
            # Para cada año, obtener el top n productos
            dfs = []
            for año in sorted(self.df['año'].unique()):
                top_año = self.df[self.df['año'] == año].groupby('producto', as_index=False)['valor_usd'].sum() \
                                                    .nlargest(top_n, 'valor_usd')
                top_año['año'] = año
                dfs.append(top_año)
            
            df_top = pd.concat(dfs)
            
            # Crear gráfico animado
            fig = px.bar(
                df_top,
                x='valor_usd',
                y='producto',
                animation_frame='año',
                orientation='h',
                title=f'<b>Evolución de los {top_n} Productos Más Exportados</b>',
                labels={'valor_usd': 'Valor Exportado (USD)', 'producto': ''},
                hover_data={'valor_usd': ':$,.0f'},
                color='valor_usd',
                color_continuous_scale='Plasma',
                range_x=[0, df_top['valor_usd'].max() * 1.1]
            )
            
            # Mejorar animación y estilo
            fig.update_traces(
                hovertemplate='<b>%{y}</b><br>Valor: %{x:$,.0f}<br>Año: %{customdata[0]}<extra></extra>'
            )
            
            fig.update_layout(
                yaxis={'categoryorder': 'total ascending'},
                coloraxis_showscale=False,
                height=650,
                transition={'duration': 1000},
                updatemenus=[{
                    'buttons': [{
                        'label': 'Play',
                        'method': 'animate',
                        'args': [None, {'frame': {'duration': 1000, 'redraw': True}}]
                    }]
                }]
            )
            
            return self._apply_style2(fig)
            
        except Exception as e:
            print(f"Error al generar gráfico animado: {str(e)}")
            return self._create_empty_figure("Error al cargar datos para animación")

    def _create_empty_figure(self, message):
        """Crea una figura vacía con mensaje de error"""
        fig = px.scatter(title=f"<b>{message}</b>")
        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        return fig