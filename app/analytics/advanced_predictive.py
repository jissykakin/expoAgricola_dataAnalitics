import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

class AdvancedPredictiveModels:
    def __init__(self, df):
        self.df = self._preprocess_data(df.copy())

        if not isinstance(df, pd.DataFrame):
            raise ValueError("El argumento debe ser un DataFrame de pandas")
            
        if df.empty:
            raise ValueError("El DataFrame no puede estar vacío")
            
        required_cols = ['fecha', 'valor_usd', 'volumen_ton']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Faltan columnas requeridas: {missing}")
        
        self.df = self._preprocess_data(df.copy())
        
    def _preprocess_data(self, df):
        """Preprocesamiento básico de los datos"""
       
        try:

            meses_espanol = {
                'enero': 1, 'febrero': 2, 'marzo': 3, 'abril': 4,
                'mayo': 5, 'junio': 6, 'julio': 7, 'agosto': 8,
                'septiembre': 9, 'octubre': 10, 'noviembre': 11, 'diciembre': 12
            }
            print(self.df)
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
            
            # df['fecha'] = pd.to_datetime(df['fecha'])
            df = df.sort_values('fecha')
            
            # Limpiar nombres de columnas
            df.columns = df.columns.str.lower().str.replace(' ', '_')
            
            # Agregar características temporales
            df['mes'] = df['fecha'].dt.month
            df['trimestre'] = df['fecha'].dt.quarter
            df['año'] = df['fecha'].dt.year

        # Validar que tenemos las columnas necesarias después de limpiar nombres
            required = ['fecha', 'valor_usd', 'volumen_ton']
            for col in required:
                if col not in df.columns:
                    raise ValueError(f"Columna requerida '{col}' no encontrada después de limpieza")
                
                return df
        except Exception as e:
                raise ValueError(f"Error en preprocesamiento: {str(e)}")
        
        # return df
    
    def prepare_time_series(self, grupo=None, depto=None):
        """Prepara la serie temporal según filtros"""
        df_filtrado = self.df.copy()
        
        if grupo:
            df_filtrado = df_filtrado[df_filtrado['tipo_producto'] == grupo]
        if depto:
            df_filtrado = df_filtrado[df_filtrado['departamento'] == depto]
        
        # Agrupar por fecha
        ts = df_filtrado.groupby('fecha').agg({
            'valor_usd': 'sum',
            'volumen_ton': 'sum'
        }).rename(columns={
            'valor_usd': 'valor',
            'volumen_ton': 'volumen'
        })
        
        return ts.asfreq('MS').fillna(0)
    
    def seasonal_decomposition(self, ts, column='valor', period=12):
        """Descomposición estacional de la serie temporal"""
        decomp = seasonal_decompose(ts[column], model='additive', period=period)
        
        # Crear figura de descomposición
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 8))
        decomp.observed.plot(ax=ax1, title='Observado')
        decomp.trend.plot(ax=ax2, title='Tendencia')
        decomp.seasonal.plot(ax=ax3, title='Estacionalidad')
        decomp.resid.plot(ax=ax4, title='Residuos')
        fig.tight_layout()
        
        return fig
    
    def test_stationarity(self, ts, column='valor'):
        """Pruebas de estacionariedad"""
        # KPSS Test
        kpss_result = kpss(ts[column].dropna())
        kpss_stat = kpss_result[0]
        kpss_pvalue = kpss_result[1]
        
        # ADF Test
        adf_result = adfuller(ts[column].dropna())
        adf_stat = adf_result[0]
        adf_pvalue = adf_result[1]
        
        return {
            'KPSS': {'Statistic': kpss_stat, 'p-value': kpss_pvalue},
            'ADF': {'Statistic': adf_stat, 'p-value': adf_pvalue}
        }
    
    def model_sarimax(self, ts, column='valor', order=(1,1,1), seasonal_order=(1,1,1,12), steps=24):
        """Modelo SARIMAX con visualización interactiva"""
        # Entrenar modelo
        model = SARIMAX(
            ts[column],
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        results = model.fit(disp=False)
        
        # Pronóstico
        forecast = results.get_forecast(steps=steps)
        conf_int = forecast.conf_int()
        
        # Crear figura interactiva con Plotly
        fig = go.Figure()
        
        # Serie observada
        fig.add_trace(go.Scatter(
            x=ts.index,
            y=ts[column],
            name='Observado',
            line=dict(color='blue')
        ))
        
        # Pronóstico
        fig.add_trace(go.Scatter(
            x=forecast.predicted_mean.index,
            y=forecast.predicted_mean,
            name='Pronóstico',
            line=dict(color='red')
        ))
        
        # Intervalo de confianza
        fig.add_trace(go.Scatter(
            x=conf_int.index,
            y=conf_int.iloc[:, 0],
            fill=None,
            mode='lines',
            line=dict(width=0),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=conf_int.index,
            y=conf_int.iloc[:, 1],
            fill='tonexty',
            mode='lines',
            line=dict(width=0),
            fillcolor='rgba(255, 0, 0, 0.2)',
            name='Intervalo 95%'
        ))
        
        fig.update_layout(
            title=f'Pronóstico SARIMAX para {column}',
            xaxis_title='Fecha',
            yaxis_title='Valor',
            hovermode='x unified'
        )
        
        return fig, results.summary()
    
    def model_prophet(self, ts, column='valor', periods=12):
        """Modelo Prophet con visualización"""
        # Preparar datos para Prophet
        df_prophet = ts.reset_index()[['fecha', column]].rename(
            columns={'fecha': 'ds', column: 'y'}
        )
        
        # Entrenar modelo
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False
        )
        model.fit(df_prophet)
        
        # Crear futuro
        future = model.make_future_dataframe(periods=periods, freq='M')
        
        # Pronóstico
        forecast = model.predict(future)
        
        # Figura interactiva
        fig = go.Figure()
        
        # Puntos observados
        fig.add_trace(go.Scatter(
            x=df_prophet['ds'],
            y=df_prophet['y'],
            name='Observado',
            mode='markers',
            marker=dict(color='blue')
        ))
        
        # Pronóstico
        fig.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat'],
            name='Pronóstico',
            line=dict(color='red')
        ))
        
        # Intervalo de confianza
        fig.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat_upper'],
            fill=None,
            mode='lines',
            line=dict(width=0),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat_lower'],
            fill='tonexty',
            mode='lines',
            line=dict(width=0),
            fillcolor='rgba(255, 0, 0, 0.2)',
            name='Intervalo 95%'
        ))
        
        fig.update_layout(
            title=f'Pronóstico Prophet para {column}',
            xaxis_title='Fecha',
            yaxis_title='Valor',
            hovermode='x unified'
        )
        
        return fig, forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    
    def model_random_forest(self, ts, column='valor', test_size=0.2, n_estimators=100):
        """Modelo Random Forest para series temporales"""
        # Crear características temporales
        df = ts.reset_index()
        df['mes'] = df['fecha'].dt.month
        df['trimestre'] = df['fecha'].dt.quarter
        df['año'] = df['fecha'].dt.year
        
        # Crear lags
        for i in [1, 2, 3, 12]:
            df[f'lag_{i}'] = df[column].shift(i)
        
        df = df.dropna()
        
        # Dividir datos
        X = df.drop(['fecha', column], axis=1)
        y = df[column]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False
        )
        
        # Modelo
        model = make_pipeline(
            StandardScaler(),
            RandomForestRegressor(
                n_estimators=n_estimators,
                random_state=42
            )
        )
        
        model.fit(X_train, y_train)
        
        # Predicciones
        y_pred = model.predict(X_test)
        
        # Figura interactiva
        fig = go.Figure()
        
        # Entrenamiento
        fig.add_trace(go.Scatter(
            x=df.iloc[:len(y_train)]['fecha'],
            y=y_train,
            name='Entrenamiento',
            line=dict(color='blue')
        ))
        
        # Test real
        fig.add_trace(go.Scatter(
            x=df.iloc[-len(y_test):]['fecha'],
            y=y_test,
            name='Test Real',
            line=dict(color='green')
        ))
        
        # Predicciones
        fig.add_trace(go.Scatter(
            x=df.iloc[-len(y_test):]['fecha'],
            y=y_pred,
            name='Predicciones',
            line=dict(color='red')
        ))
        
        fig.update_layout(
            title=f'Predicción Random Forest para {column}',
            xaxis_title='Fecha',
            yaxis_title='Valor',
            hovermode='x unified'
        )
        
        # Métricas
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        return fig, {'MAE': mae, 'RMSE': rmse}
    
    def model_svm(self, ts, column='valor', test_size=0.2, kernel='rbf'):
        """Modelo SVM para series temporales"""
        # Crear características (similar a Random Forest)
        df = ts.reset_index()
        df['mes'] = df['fecha'].dt.month
        df['trimestre'] = df['fecha'].dt.quarter
        df['año'] = df['fecha'].dt.year
        
        for i in [1, 2, 3, 12]:
            df[f'lag_{i}'] = df[column].shift(i)
        
        df = df.dropna()
        
        # Dividir datos
        X = df.drop(['fecha', column], axis=1)
        y = df[column]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False
        )
        
        # Modelo SVM
        model = make_pipeline(
            StandardScaler(),
            SVR(kernel=kernel)
        )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Figura interactiva
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df.iloc[:len(y_train)]['fecha'],
            y=y_train,
            name='Entrenamiento',
            line=dict(color='blue')
        ))
        
        fig.add_trace(go.Scatter(
            x=df.iloc[-len(y_test):]['fecha'],
            y=y_test,
            name='Test Real',
            line=dict(color='green')
        ))
        
        fig.add_trace(go.Scatter(
            x=df.iloc[-len(y_test):]['fecha'],
            y=y_pred,
            name='Predicciones SVM',
            line=dict(color='purple')
        ))
        
        fig.update_layout(
            title=f'Predicción SVM ({kernel}) para {column}',
            xaxis_title='Fecha',
            yaxis_title='Valor',
            hovermode='x unified'
        )
        
        # Métricas
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        return fig, {'MAE': mae, 'RMSE': rmse}