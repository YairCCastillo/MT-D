from shiny.express import input, render, ui
import pandas as pd
import plotly.express as px
from shinywidgets import render_plotly
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA

mtcars = pd.read_csv('mtcars.csv')
mpg_series = pd.Series(mtcars['mpg'].values, 
            index=pd.date_range(start='2023-01-01', periods=len(mtcars), freq='ME'))
# Perform time series decomposition
decomposition = seasonal_decompose(mpg_series, model='additive')

arima_model = ARIMA(mpg_series, order=(6, 0,2))
arima_fit = arima_model.fit()

# Forecast the next 6 months
forecast = arima_fit.forecast(steps=6)
ui.page_opts(title = "Pronósticos")
with ui.sidebar():
   ui.input_numeric("pp", "p", 8)
   ui.input_numeric("dd", "d", 1)
   ui.input_numeric("qq", "q", 1)
   
   
   
   
   

with ui.layout_columns(col_widths=[6, 6, 12]):
    with ui.card(full_screen=True):
        ui.card_header("Gráficos de la descomposición")
        @render_plotly
        def observed_plot():
        # Create the observed component plot
            fig = px.line(
            x=decomposition.observed.index,
            y=decomposition.observed,
            labels={"x": "Fecha", "y": "mpo"},
            title="Componente Observado"
            )
            return fig
    with ui.card(full_screen=False):
        @render_plotly
        def trend_plot():
        # Create the trend component plot
            fig = px.line(
            x=decomposition.trend.index,
            y=decomposition.trend,
            labels={"x": "Fecha", "y": "Tendencia"},
            title="Componente de tendencia"
            )
            return fig
    with ui.card(full_screen=False):
        @render_plotly
        def seasonal_plot():
            # Create the seasonal component plot
            fig = px.line(
                x=decomposition.seasonal.index,
                y=decomposition.seasonal,
                labels={"x": "Fecha", "y": "Estacionalidad"},
                title="Componente estacional"
            )
            return fig
    with ui.card(full_screen=False):
        @render_plotly
        def residual_plot():
            # Create the residual component plot
            fig = px.line(
                x=decomposition.resid.index,
                y=decomposition.resid,
                labels={"x": "Fecha", "y": "Residuos"},
                title="Componente residual"
            )
            return fig
    with ui.card(full_screen=True):
        ui.card_header("Predicción con ARIMA")
        @render_plotly
        def arima_plot():
            # Graficar los datos originales y la predicción
            fig = px.line(
                mpg_series,
                #x=pd.DataFrame(mpg_series).index,
                #y="Observed",
                title="Predicción ARIMA para MPG",
                labels={"Observed": "MPG", "Date": "Date"}
            )
            # Destacar la predicción
            fig.add_scatter(
                x=forecast.index, 
                y=forecast, 
                mode='lines+markers', 
                name='Forecast', 
                line=dict(dash='dot')
            )
            return fig
    with ui.card(full_screen=True):
        @render_plotly
        def arima_plot2():
            arima_model2 = ARIMA(mpg_series, order=(input.pp(),input.dd(),input.qq()))
            arima_fit2 = arima_model2.fit()

            # Forecast the next 6 months
            forecast2 = arima_fit2.forecast(steps=6)
            # Graficar los datos originales y la predicción
            fig = px.line(
                mpg_series,
                #x=pd.DataFrame(mpg_series).index,
                #y="Observed",
                title="Predicción ARIMA para MPG",
                labels={"Observed": "MPG", "Date": "Date"}
            )
            # Destacar la predicción
            fig.add_scatter(
                x=forecast2.index, 
                y=forecast2, 
                mode='lines+markers', 
                name='Forecast', 
                line=dict(dash='dot')
            )
            return fig