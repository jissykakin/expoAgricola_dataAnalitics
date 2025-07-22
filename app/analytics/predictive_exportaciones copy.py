import pandas as pd
import numpy as np
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime, timedelta

class PredictiveExportaciones:
    def __init__(self, df):
        self.df = df.copy()
        self._preparar_datos()
    
    def _preparar_datos(self):
        """Prepara los datos para modelado"""

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
        self.df['mes_num'] = self.df['fecha'].dt.month
        self.df['trimestre'] = self.df['fecha'].dt.quarter
        self.df['año'] = self.df['fecha'].dt.year
        
        # Limpiar nombres de columnas
        self.df.columns = self.df.columns.str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
        print("df_inicial", self.df)

    def _apply_style(self, fig):
        """Aplica estilos consistentes a los gráficos"""
        fig.update_layout(
            margin=dict(l=50, r=50, t=80, b=50),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Arial", size=12),
            hoverlabel=dict(bgcolor="#2a3f5f", font_size=12)
        )
        return fig
    
    def predecir(self, producto=None, departamento=None, modelo='arima', horizonte=12):
        """Genera predicciones según los filtros"""
        """Genera predicciones con validación de datos"""
        # Filtrar datos
        df_filtrado = self.df.copy()
        if producto:
            df_filtrado = df_filtrado[df_filtrado['producto'] == producto]
        if departamento:
            df_filtrado = df_filtrado[df_filtrado['departamento'] == departamento]
        
        if len(df_filtrado) < 12:
            empty_df = pd.DataFrame(columns=['fecha', 'valor_usd', 
                                        'volumen_ton'])
            return {
                'valor': {
                    'predicciones': empty_df,
                    'metricas': {'error': 'Insuficientes datos'}
                },
                'volumen': {
                    'predicciones': empty_df,
                    'metricas': {'error': 'Insuficientes datos'}
                },
                'historico': empty_df
            }
   
        # Agrupar por fecha
        df_agrupado = df_filtrado.groupby('fecha', as_index=False).agg({
            'valor_usd': 'sum',
            'volumen_ton': 'sum'
        }).sort_values('fecha')

         # Asegurar que tenemos columnas necesarias
        required_cols = ['fecha', 'valor_usd', 'volumen_ton']
        for col in required_cols:
            if col not in df_agrupado.columns:
                df_agrupado[col] = 0  # O manejar el error adecuadamente
        
        
        # Preparar series temporales
        df_agrupado = df_agrupado.set_index('fecha').asfreq('MS').fillna(0).reset_index()
        
        # Dividir en train/test
        train_size = int(len(df_agrupado) * 0.8)
        train, test = df_agrupado.iloc[:train_size], df_agrupado.iloc[train_size:]
        
        # Modelado según selección
        if modelo == 'arima':
            pred_valor, met_valor = self._modelar_arima(train, test, 'valor_usd', horizonte)
            pred_volumen, met_volumen = self._modelar_arima(train, test, 'volumen_ton', horizonte)
        elif modelo == 'prophet':
            pred_valor, met_valor = self._modelar_prophet(train, test, 'valor_usd', horizonte)
            pred_volumen, met_volumen = self._modelar_prophet(train, test, 'volumen_ton', horizonte)
        elif modelo == 'rf':
            pred_valor, met_valor = self._modelar_rf(train, test, 'valor_usd', horizonte)
            pred_volumen, met_volumen = self._modelar_rf(train, test, 'volumen_ton', horizonte)
        
        # Preparar resultados
        resultados = {
            'valor': {
                'predicciones': pred_valor,
                'metricas': met_valor
            },
            'volumen': {
                'predicciones': pred_volumen,
                'metricas': met_volumen
            },
            'historico': df_agrupado
        }
        
        return resultados
    
    def _modelar_arima(self, train, test, columna, horizonte):
        """Modelo ARIMA mejorado con manejo de errores robusto"""
        try:
            # Verificar que tenemos suficientes datos
            if len(train) < 12:  # Mínimo 12 observaciones (1 año)
                raise ValueError("Insuficientes datos para modelar ARIMA (mínimo 12 observaciones)")
            
            # Asegurar que la serie temporal está completa
            train_series = train.set_index('fecha')[columna].asfreq('MS').fillna(method='ffill')
            
            # Determinar el orden del modelo ARIMA
            # Podemos usar auto_arima en producción, pero para el ejemplo usaremos (1,1,1)
            order = (1, 1, 1)  # (p,d,q)
            
            # Entrenar modelo con configuración robusta
            model = ARIMA(train_series, order=order)
            model_fit = model.fit(method_kwargs={'warn_convergence': False})
            
            # Predecir
            if len(test) > 0:
                predictions = model_fit.forecast(steps=len(test))
                pred_df = test[['fecha', columna]].copy()
                pred_df['prediccion'] = predictions.values
                pred_df['tipo'] = 'test'
            else:
                # Predecir futuro
                predictions = model_fit.forecast(steps=horizonte)
                future_dates = pd.date_range(
                    start=train_series.index[-1] + pd.offsets.MonthBegin(1),
                    periods=horizonte,
                    freq='MS'
                )
                pred_df = pd.DataFrame({
                    'fecha': future_dates,
                    columna: [np.nan] * horizonte,
                    'prediccion': predictions.values,
                    'tipo': 'future'
                })
            
            # Calcular métricas si hay datos de test
            metricas = {}
            if len(test) > 0:
                y_true = test[columna].values
                y_pred = predictions.values
                metricas['mae'] = mean_absolute_error(y_true, y_pred)
                metricas['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
                metricas['error_pct'] = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            
            return pred_df, metricas
            
        except Exception as e:
            print(f"Error en modelo ARIMA mejorado: {str(e)}")
            # Devolver DataFrame vacío con estructura correcta
            empty_df = pd.DataFrame(columns=['fecha', columna, 'prediccion', 'tipo'])
            return empty_df, {'error': str(e)}
    
    def _modelar_prophet(self, train, test, columna, horizonte):
        """Modelo Prophet para series temporales"""
        try:
            # Preparar datos para Prophet
            df_prophet = train[['fecha', columna]].rename(columns={'fecha': 'ds', columna: 'y'})
            
            # Entrenar modelo
            model = Prophet()
            model.fit(df_prophet)
            
            # Crear DataFrame futuro
            future = model.make_future_dataframe(periods=len(test) if len(test) > 0 else model.make_future_dataframe(periods=horizonte, freq='M'))
            
            # Predecir
            forecast = model.predict(future)
            
            # Preparar resultados
            pred_df = forecast[['ds', 'yhat']].rename(columns={'ds': 'fecha', 'yhat': 'prediccion'})
            pred_df[columna] = pd.concat([train[columna], test[columna]] if len(test) > 0 else train[columna]).values
            pred_df['tipo'] = 'test' if len(test) > 0 else 'future'
            
            # Calcular métricas
            metricas = {}
            if len(test) > 0:
                y_true = test[columna].values
                y_pred = forecast.iloc[-len(test):]['yhat'].values
                metricas['mae'] = mean_absolute_error(y_true, y_pred)
                metricas['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
            
            return pred_df, metricas
            
        except Exception as e:
            print(f"Error en modelo Prophet: {str(e)}")
            return None, {}
    
    def _modelar_rf(self, train, test, columna, horizonte):
        """Modelo Random Forest para series temporales"""
        try:
            # Crear características temporales
            train['mes'] = train['fecha'].dt.month
            train['trimestre'] = train['fecha'].dt.quarter
            train['año'] = train['fecha'].dt.year
            
            # Entrenar modelo
            X_train = train[['mes', 'trimestre', 'año']]
            y_train = train[columna]
            
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Predecir
            if len(test) > 0:
                test['mes'] = test['fecha'].dt.month
                test['trimestre'] = test['fecha'].dt.quarter
                test['año'] = test['fecha'].dt.year
                X_test = test[['mes', 'trimestre', 'año']]
                predictions = model.predict(X_test)
                
                pred_df = test[['fecha', columna]].copy()
                pred_df['prediccion'] = predictions
                pred_df['tipo'] = 'test'
            else:
                # Generar fechas futuras
                last_date = train['fecha'].iloc[-1]
                future_dates = [last_date + timedelta(days=30*i) for i in range(1, horizonte+1)]
                
                # Preparar características
                future_df = pd.DataFrame({'fecha': future_dates})
                future_df['mes'] = future_df['fecha'].dt.month
                future_df['trimestre'] = future_df['fecha'].dt.quarter
                future_df['año'] = future_df['fecha'].dt.year
                
                # Predecir
                predictions = model.predict(future_df[['mes', 'trimestre', 'año']])
                
                pred_df = pd.DataFrame({'fecha': future_dates})
                pred_df[columna] = [np.nan]*horizonte
                pred_df['prediccion'] = predictions
                pred_df['tipo'] = 'future'
            
            # Calcular métricas
            metricas = {}
            if len(test) > 0:
                metricas['mae'] = mean_absolute_error(test[columna], predictions)
                metricas['rmse'] = np.sqrt(mean_squared_error(test[columna], predictions))
            
            return pred_df, metricas
            
        except Exception as e:
            print(f"Error en modelo Random Forest: {str(e)}")
            return None, {}
    
    def generar_grafico_prediccion(self, df_historico, df_prediccion, columna, titulo):
        """Genera gráfico interactivo de predicción"""
        if df_prediccion is None:
            return self._create_empty_figure("No hay datos de predicción")
        
        # Combinar histórico y predicción
        df_plot = pd.concat([
            df_historico[['fecha', columna]].assign(tipo='histórico'),
            df_prediccion[['fecha', 'prediccion', 'tipo']].rename(columns={'prediccion': columna})
        ])
        
        # Crear gráfico
        fig = px.line(
            df_plot,
            x='fecha',
            y=columna,
            color='tipo',
            title=f'<b>{titulo}</b>',
            labels={'fecha': 'Fecha', columna: 'Valor'},
            color_discrete_map={
                'histórico': '#1f77b4',
                'test': '#ff7f0e',
                'future': '#2ca02c'
            }
        )
        
        # Mejorar estilo
        fig.update_traces(
            line_width=2.5,
            hovertemplate='Fecha: %{x|%b %Y}<br>Valor: %{y:,.0f}<extra></extra>'
        )
        
        fig.update_layout(
            hovermode='x unified',
            legend_title_text='Datos',
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(step="all")
                    ])
                ),
                rangeslider=dict(visible=True),
                type="date"
            )
        )
        
        return self._apply_style(fig)
    
    def _create_empty_figure(self, message):
        """Crea una figura vacía con mensaje"""
        fig = px.scatter(title=f"<b>{message}</b>")
        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        return fig