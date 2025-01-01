import pandas as pd
import faicons as fa
from shiny.express import input, render, ui
from shinywidgets import render_plotly
import plotly.express as px

ui.page_opts(title="HeatMap", fillable=True)

ICONS = {
    "user": fa.icon_svg("diamond"),
    "wallet": fa.icon_svg("wallet"),
    "currency-dollar": fa.icon_svg("dollar-sign"),
    "gear": fa.icon_svg("gear")
}

with ui.sidebar():
   ui.input_numeric("pric", "Precio del diamante", 5000)
   
diamante = pd.read_csv('diamonds.csv')
diamante['price_per_carat'] = diamante['price'] / diamante['carat']
heatmap_data = diamante.groupby(['cut', 'color'])['price_per_carat'].mean().reset_index()
heatmap_pivot = heatmap_data.pivot(index='cut', columns='color', values='price_per_carat')

with ui.layout_columns(fill=False):

    with ui.value_box(showcase=ICONS["user"]):
        "Probabilidad de encontrar un diamante más caro que el precio dado"
        @render.express
        def total_tippers():
            prob_cost_may(input.pric())

with ui.card(full_screen=False):
    ui.card_header("Heatmap")
    @render_plotly
    def heatmap_plot():
        # Crear el heatmap con plotly
        fig = px.imshow(
            heatmap_pivot,
            labels=dict(x="Color", y="Corte", color="Precio Promedio por Quilate"),
            x=heatmap_pivot.columns,
            y=heatmap_pivot.index,
            color_continuous_scale="Viridis"
        )
        fig.update_layout(title="Precio Promedio por Quilate (por Corte y Color)", title_x=0.5)
        return fig

def prob_cost_may(precio):
    total_diamantes = len(diamante)
    check_mayor = len(diamante[diamante['price'] > precio])
    return check_mayor / total_diamantes