import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
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
    
    # def predecir(self, producto=None, departamento=None, modelo='sarima', horizonte=12):
    #     """Genera predicciones según los filtros"""
    #     """Genera predicciones con validación de datos"""
    #     # Filtrar datos
    #     df_filtrado = self.df.copy()
    #     if producto:
    #         df_filtrado = df_filtrado[df_filtrado['producto'] == producto]
    #     if departamento:
    #         df_filtrado = df_filtrado[df_filtrado['departamento'] == departamento]
        
    #     if len(df_filtrado) < 12:
    #         empty_df = pd.DataFrame(columns=['fecha', 'valor_usd', 
    #                                     'volumen_ton'])
    #         return {
    #             'valor': {
    #                 'predicciones': empty_df,
    #                 'metricas': {'error': 'Insuficientes datos'}
    #             },
    #             'volumen': {
    #                 'predicciones': empty_df,
    #                 'metricas': {'error': 'Insuficientes datos'}
    #             },
    #             'historico': empty_df
    #         }
   
    #     # Agrupar por fecha
    #     df_agrupado = df_filtrado.groupby('fecha', as_index=False).agg({
    #         'valor_usd': 'sum',
    #         'volumen_ton': 'sum'
    #     }).sort_values('fecha')

    #      # Asegurar que tenemos columnas necesarias
    #     required_cols = ['fecha', 'valor_usd', 'volumen_ton']
    #     for col in required_cols:
    #         if col not in df_agrupado.columns:
    #             df_agrupado[col] = 0  # O manejar el error adecuadamente
        
        
    #     # Preparar series temporales
    #     df_agrupado = df_agrupado.set_index('fecha').asfreq('MS').fillna(0).reset_index()
        
    #     # Dividir en train/test
    #     train_size = int(len(df_agrupado) * 0.8)
    #     train, test = df_agrupado.iloc[:train_size], df_agrupado.iloc[train_size:]
        
    #     # Modelado según selección
    #     if modelo == 'sarima':
    #         pred_valor, met_valor = self._modelar_sarima(train, test, 'valor_usd', horizonte)
    #         pred_volumen, met_volumen = self._modelar_sarima(train, test, 'volumen_ton', horizonte)
    #     elif modelo == 'prophet':
    #         pred_valor, met_valor = self._modelar_prophet(train, test, 'valor_usd', horizonte)
    #         pred_volumen, met_volumen = self._modelar_prophet(train, test, 'volumen_ton', horizonte)
    #     elif modelo == 'rf':
    #         pred_valor, met_valor = self._modelar_rf(train, test, 'valor_usd', horizonte)
    #         pred_volumen, met_volumen = self._modelar_rf(train, test, 'volumen_ton', horizonte)
        
    #     # Preparar resultados
    #     resultados = {
    #         'valor': {
    #             'predicciones': pred_valor,
    #             'metricas': met_valor
    #         },
    #         'volumen': {
    #             'predicciones': pred_volumen,
    #             'metricas': met_volumen
    #         },
    #         'historico': df_agrupado
    #     }
        
    #     return resultados

    def predecir(self, producto=None, departamento=None, modelo='sarima', horizonte=12):
        """Genera predicciones con manejo robusto de errores"""
        try:
            # Inicializar variables con valores por defecto
            pred_valor = pd.DataFrame(columns=['fecha', 'valor_usd', 'prediccion', 'tipo'])
            met_valor = {'error': 'No se pudo calcular'}
            pred_volumen = pd.DataFrame(columns=['fecha', 'volumen_ton', 'prediccion', 'tipo'])
            met_volumen = {'error': 'No se pudo calcular'}
            
            # Filtrar datos
            df_filtrado = self.df.copy()
            if producto:
                df_filtrado = df_filtrado[df_filtrado['producto'] == producto]
            if departamento:
                df_filtrado = df_filtrado[df_filtrado['departamento'] == departamento]
            
            if len(df_filtrado) < 24:  # Mínimo para SARIMA
                raise ValueError("Insuficientes datos para modelar (mínimo 24 observaciones)")
            
            # Agrupar por fecha
            df_agrupado = df_filtrado.groupby('fecha', as_index=False).agg({
                'valor_usd': 'sum',
                'volumen_ton': 'sum'
            }).sort_values('fecha').set_index('fecha').asfreq('MS').fillna(0).reset_index()
            
            # Dividir train/test
            train_size = int(len(df_agrupado) * 0.8)
            train, test = df_agrupado.iloc[:train_size], df_agrupado.iloc[train_size:]
            
            # Modelado según selección
            if modelo == 'sarima':
                pred_valor, met_valor = self._modelar_sarima(train, test, 'valor_usd', horizonte)
                pred_volumen, met_volumen = self._modelar_sarima(train, test, 'volumen_ton', horizonte)
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
                    'metricas': met_valor,
                     
                },
                'volumen': {
                    'predicciones': pred_volumen,
                    'metricas': met_volumen,
                    'diagnosticos': met_volumen.get('diagnosticos', None)
                },
                'historico': df_agrupado
            }
            
            return resultados
            
        except Exception as e:
            print(f"Error en predicción: {str(e)}")
            # Devolver DataFrames vacíos pero con estructura consistente
            empty_df = pd.DataFrame(columns=['fecha', 'valor_usd', 'volumen_ton', 'prediccion', 'tipo'])
            return {
                'valor': {
                    'predicciones': empty_df,
                    'metricas': {'error': str(e)},
                    'diagnosticos': None
                },
                'volumen': {
                    'predicciones': empty_df,
                    'metricas': {'error': str(e)},
                    'diagnosticos': None
                },
                'historico': empty_df
            }
    
    # def _modelar_sarima(self, train, test, columna, horizonte):
    #     """Modelo SARIMA con manejo robusto de errores y diagnósticos"""
    #     try:
    #         # Verificar datos mínimos
    #         if len(train) < 24:
    #             raise ValueError("Insuficientes datos para SARIMA (mínimo 24 meses)")
            
    #         train_series = train.set_index('fecha')[columna].asfreq('MS').fillna(method='ffill')
            
    #         # Configuración SARIMA - puedes ajustar estos parámetros
    #         order = (1, 1, 1)
    #         seasonal_order = (1, 1, 1, 12)
            
    #         # Entrenar modelo
    #         model = SARIMAX(train_series,
    #                     order=order,
    #                     seasonal_order=seasonal_order,
    #                     enforce_stationarity=False,
    #                     enforce_invertibility=False)
            
    #         model_fit = model.fit(disp=False)
            
    #         # Generar diagnósticos (pero no devolverlos aquí)
    #         fig_diagnosticos = self._generar_diagnosticos_sarima(model_fit, train_series)

            
    #         # Predicciones
    #         pred_df = pd.DataFrame()
            
    #         # 1. Predicciones para test (si existe)
    #         if len(test) > 0:
    #             test_pred = model_fit.get_forecast(steps=len(test))
    #             test_df = test[['fecha', columna]].copy()
    #             test_df['prediccion'] = test_pred.predicted_mean.values
    #             test_df['tipo'] = 'test'
    #             pred_df = pd.concat([pred_df, test_df])
            
    #         # 2. Predicciones futuras
    #         future_steps = horizonte if len(test) == 0 else horizonte - len(test)
    #         if future_steps > 0:
    #             future_pred = model_fit.get_forecast(steps=future_steps)
    #             future_dates = pd.date_range(
    #                 start=train_series.index[-1] if len(test) == 0 else test['fecha'].iloc[-1],
    #                 periods=future_steps,
    #                 freq='MS'
    #             )
    #             future_df = pd.DataFrame({
    #                 'fecha': future_dates,
    #                 columna: np.nan,
    #                 'prediccion': future_pred.predicted_mean.values,
    #                 'tipo': 'future'
    #             })
    #             pred_df = pd.concat([pred_df, future_df])
            
    #         # Métricas
    #         metricas = {}
    #         if len(test) > 0:
    #             y_true = test[columna].values
    #             y_pred = test_pred.predicted_mean.values
    #             metricas.update({
    #                 'mae': mean_absolute_error(y_true, y_pred),
    #                 'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
    #                 'error_pct': np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100,
                    
    #             })

    #             print("diagnostico_print", fig_diagnosticos)
            
    #         metricas['diagnosticos'] = fig_diagnosticos

               
            
    #     except Exception as e:
    #         print(f"Error en SARIMA para {columna}: {str(e)}")
    #         empty_df = pd.DataFrame(columns=['fecha', columna, 'prediccion', 'tipo'])
    #         return empty_df, {'error': str(e)}

    
    def _modelar_sarima(self, train, test, columna, horizonte):
        """Modelo SARIMA corregido para predecir desde datos históricos"""
        try:
            # Verificar datos mínimos
            if len(train) < 24:
                raise ValueError("Insuficientes datos para SARIMA (mínimo 24 meses)")
            
            # Preparar serie temporal
            train_series = train.set_index('fecha')[columna].asfreq('MS').fillna(method='ffill')
            
            # Configuración SARIMA - puedes optimizar estos parámetros
            order = (1, 1, 1)
            seasonal_order = (1, 1, 1, 12)
            
            # Entrenar modelo
            model = SARIMAX(train_series,
                        order=order,
                        seasonal_order=seasonal_order,
                        enforce_stationarity=False,
                        enforce_invertibility=False)
            
            model_fit = model.fit(disp=False)
            
            # Generar diagnósticos
            fig_diagnosticos = self._generar_diagnosticos_sarima(model_fit, train_series)
            
            # DataFrame para resultados
            pred_df = pd.DataFrame()
            
            # 1. Predicciones para el período completo (train + test + futuro)
            total_steps = len(test) + horizonte if len(test) > 0 else horizonte
            predictions = model_fit.get_forecast(steps=total_steps)
            
            # Fechas para las predicciones
            start_date = train_series.index[-1] + pd.offsets.MonthBegin(1)
            pred_dates = pd.date_range(start=start_date, periods=total_steps, freq='MS')
            
            # Crear DataFrame con todas las predicciones
            full_pred_df = pd.DataFrame({
                'fecha': pred_dates,
                columna: np.nan,
                'prediccion': predictions.predicted_mean.values,
                'tipo': 'future'  # Marcamos todo como futuro inicialmente
            })
            
            # 2. Marcar el período de test si existe
            if len(test) > 0:
                test_dates = test['fecha'].values
                full_pred_df.loc[full_pred_df['fecha'].isin(test_dates), 'tipo'] = 'test'
                # Agregar valores reales del test
                full_pred_df = pd.merge(full_pred_df, 
                                    test[['fecha', columna]], 
                                    on='fecha', 
                                    how='left',
                                    suffixes=('', '_real'))
                full_pred_df[columna] = full_pred_df[columna+'_real'].combine_first(full_pred_df[columna])
                full_pred_df.drop(columns=[columna+'_real'], inplace=True)
            
            # 3. Seleccionar solo el horizonte requerido
            pred_df = full_pred_df.head(horizonte + len(test)) if len(test) > 0 else full_pred_df.head(horizonte)
            
            # Calcular métricas si hay test
            metricas = {'diagnosticos': fig_diagnosticos}
            if len(test) > 0:
                test_period = pred_df[pred_df['tipo'] == 'test']
                if not test_period.empty:
                    y_true = test_period[columna].values
                    y_pred = test_period['prediccion'].values
                    metricas.update({
                        'mae': mean_absolute_error(y_true, y_pred),
                        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                        'error_pct': np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
                    })
            
            return pred_df, metricas
            
        except Exception as e:
            print(f"Error en SARIMA para {columna}: {str(e)}")
            empty_df = pd.DataFrame(columns=['fecha', columna, 'prediccion', 'tipo'])
            return empty_df, {'error': str(e)}
            
    def _modelar_prophet(self, train, test, columna, horizonte):
        try:
            # Preparar datos para Prophet
            df_prophet = train[['fecha', columna]].rename(columns={'fecha': 'ds', columna: 'y'})
            
            # Entrenar modelo
            model = Prophet()
            model.fit(df_prophet)
            
            # Crear DataFrame futuro
            future = model.make_future_dataframe(periods=horizonte, freq='MS')
            
            # Predecir
            forecast = model.predict(future)
            
            # Preparar DataFrame de resultados
            pred_df = forecast[['ds', 'yhat']].rename(columns={
                'ds': 'fecha',
                'yhat': 'prediccion'
            })
            
            # Solo mantener las predicciones futuras (no duplicar históricos)
            pred_df = pred_df[~pred_df['fecha'].isin(train['fecha'])]  # Eliminar datos de entrenamiento
            
            # Marcar tipo de predicción
            pred_df['tipo'] = 'future'
            pred_df[columna] = np.nan  # Valores reales son desconocidos para el futuro
            
            # Si hay datos de test, marcarlos
            if len(test) > 0:
                test_dates = test['fecha'].values
                pred_df.loc[pred_df['fecha'].isin(test_dates), 'tipo'] = 'test'
                # Agregar valores reales del test
                for fecha in test_dates:
                    pred_df.loc[pred_df['fecha'] == fecha, columna] = test[test['fecha'] == fecha][columna].values[0]
            
            # Calcular métricas si hay test
            metricas = {}
            if len(test) > 0:
                test_mask = pred_df['tipo'] == 'test'
                y_true = pred_df.loc[test_mask, columna].values
                y_pred = pred_df.loc[test_mask, 'prediccion'].values
                metricas = {
                    'mae': mean_absolute_error(y_true, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y_true, y_pred))
                }
            
            return pred_df, metricas
            
        except Exception as e:
            print(f"Error en modelo Prophet: {str(e)}")
            empty_df = pd.DataFrame(columns=['fecha', columna, 'prediccion', 'tipo'])
            return empty_df, {'error': str(e)}
    
    def _modelar_rf(self, train, test, columna, horizonte):
        """Modelo Random Forest corregido con características temporales mejoradas"""
        try:
            # Crear características temporales más robustas
            def create_features(df):
                df = df.copy()
                df['mes'] = df['fecha'].dt.month
                df['trimestre'] = df['fecha'].dt.quarter
                df['año'] = df['fecha'].dt.year
                df['dias_mes'] = df['fecha'].dt.days_in_month
                df['dias_año'] = df['fecha'].dt.dayofyear
                df['semana_año'] = df['fecha'].dt.isocalendar().week
                return df
            
            train = create_features(train)
            
            # Entrenar modelo
            features = ['mes', 'trimestre', 'año', 'dias_mes', 'dias_año', 'semana_año']
            X_train = train[features]
            y_train = train[columna]
            
            model = RandomForestRegressor(
                n_estimators=200,
                random_state=42,
                min_samples_split=5,
                max_depth=10
            )
            model.fit(X_train, y_train)
            
            # DataFrame para resultados
            pred_df = pd.DataFrame()
            
            # 1. Predicciones para período de test (si existe)
            if len(test) > 0:
                test = create_features(test)
                X_test = test[features]
                test_predictions = model.predict(X_test)
                
                test_df = test[['fecha', columna]].copy()
                test_df['prediccion'] = test_predictions
                test_df['tipo'] = 'test'
                pred_df = pd.concat([pred_df, test_df])
            
            # 2. Predicciones futuras (siempre generamos)
            future_steps = horizonte if len(test) == 0 else horizonte - len(test)
            if future_steps > 0:
                # Generar fechas futuras
                last_date = train['fecha'].iloc[-1] if len(test) == 0 else test['fecha'].iloc[-1]
                future_dates = pd.date_range(
                    start=last_date + pd.offsets.MonthBegin(1),
                    periods=future_steps,
                    freq='MS'
                )
                
                # Crear DataFrame futuro con características
                future_df = pd.DataFrame({'fecha': future_dates})
                future_df = create_features(future_df)
                future_predictions = model.predict(future_df[features])
                
                # Preparar resultados
                future_df[columna] = np.nan
                future_df['prediccion'] = future_predictions
                future_df['tipo'] = 'future'
                future_df = future_df[['fecha', columna, 'prediccion', 'tipo']]
                
                pred_df = pd.concat([pred_df, future_df])
            
            # Calcular métricas para test
            metricas = {}
            if len(test) > 0:
                y_true = test[columna].values
                y_pred = test_predictions
                metricas['mae'] = mean_absolute_error(y_true, y_pred)
                metricas['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
                metricas['error_pct'] = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
            
            return pred_df.sort_values('fecha'), metricas
            
        except Exception as e:
            print(f"Error en modelo Random Forest: {str(e)}")
            empty_df = pd.DataFrame(columns=['fecha', columna, 'prediccion', 'tipo'])
            return empty_df, {'error': str(e)}
    
    def _generar_diagnosticos_sarima(self, model_fit, train_series):
        """Versión robusta para generar diagnósticos SARIMA"""
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            # Crear figura Plotly directamente (sin pasar por matplotlib)
            fig = make_subplots(rows=2, cols=1, 
                            subplot_titles=("Diagnósticos SARIMA", "Descomposición Estacional"),
                            vertical_spacing=0.15)
            
            # 1. Gráficos de diagnóstico SARIMA (simulados)
            residuals = model_fit.resid.dropna()
            fig.add_histogram(x=residuals, row=1, col=1, name="Residuos")
            
            # 2. Descomposición estacional
            decomposition = seasonal_decompose(train_series, model='additive', period=12)
            
            # Añadir cada componente
            fig.add_scatter(x=train_series.index, y=decomposition.observed, 
                        mode='lines', name='Observado', row=2, col=1)
            fig.add_scatter(x=train_series.index, y=decomposition.trend, 
                        mode='lines', name='Tendencia', row=2, col=1)
            fig.add_scatter(x=train_series.index, y=decomposition.seasonal, 
                        mode='lines', name='Estacionalidad', row=2, col=1)
            fig.add_scatter(x=train_series.index, y=decomposition.resid, 
                        mode='lines', name='Residuos', row=2, col=1)
            
            # Ajustar layout
            fig.update_layout(
                height=800,
                showlegend=True,
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            
            return fig
            
        except Exception as e:
            print(f"Error generando diagnósticos: {str(e)}")
            return self._create_empty_plotly_figure(f"Error en diagnósticos: {str(e)}")
    
    def mostrar_diagnosticos_sarima(self, resultados, variable='valor'):
        """Versión simplificada que devuelve figuras Plotly directamente"""

        print(variable)
        print(resultados)

        try:
            if not resultados or variable not in resultados:
                return self._create_empty_plotly_figure("No hay resultados disponibles")
            
             
            # Ahora devolvemos directamente la figura Plotly
            return resultados[variable].get('metricas', {}).get('diagnosticos', 
                self._create_empty_plotly_figure("No hay gráficos disponibles"))
            
        except Exception as e:
            print(f"Error mostrando diagnósticos: {str(e)}")
            return self._create_empty_plotly_figure(f"Error: {str(e)}")

    def _create_empty_plotly_figure(self, message):
            """Crea una figura Plotly vacía con mensaje de error"""
            import plotly.graph_objects as go  # Import aquí si no lo tienes global
            
            fig = go.Figure()
            fig.add_annotation(
                text=message,
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16))
            fig.update_layout(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            return fig
    def generar_grafico_prediccion(self, df_historico, df_prediccion, columna, titulo):
        """Genera gráfico interactivo de predicción"""
        if df_prediccion is None or len(df_prediccion) == 0:
            return self._create_empty_figure("No hay datos de predicción")
        
        # Crear DataFrame para el gráfico
        df_plot = df_historico[['fecha', columna]].copy()
        df_plot['tipo'] = 'histórico'
        
        # Agregar predicciones
        df_pred = df_prediccion[['fecha', 'prediccion', 'tipo']].copy()
        df_pred = df_pred.rename(columns={'prediccion': columna})
        
        df_plot = pd.concat([df_plot, df_pred])
        
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