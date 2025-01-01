from shiny.express import input, render, ui
import pandas as pd
import faicons as fa
from shinywidgets import render_plotly
import plotly.express as px

ICONS = {
    "currency-dollar": fa.icon_svg("dollar-sign"),
}

diamante = pd.read_csv('diamonds.csv')
mtcars = pd.read_csv('mtcars.csv')
mtcars['price'] = mtcars['mpg']*1000

with ui.sidebar():
   ui.input_numeric("pric", "Precio del diamante", 30000)

with ui.layout_columns(col_widths=[6, 6, 12]):
    with ui.card(full_screen=True):
        ui.card_header("Mejor Carro con el presupuestp")
        @render.data_frame
        def table():
            return mejor_compra_carro(input.pric())
        
    with ui.card(full_screen=True):
        ui.card_header("Mejor diamante con el presupuesto")
        @render.data_frame
        def table2():
            return mejor_compra_diamante(input.pric())
        
      
    with ui.value_box(showcase=ICONS["currency-dollar"]):
         "Suma de presupuestos del mejor carro y el mejor diamante"
         @render.express
         def total_tippers():
             mejor_compra_combinada(input.pric())
                        
    with ui.card(full_screen=True):
        ui.card_header("Distribución de presupuesto")
        @render_plotly         
        def plot():
            data = {"Categoria": ["Carro", "Diamante"],
                    "Valores": mejor_compra(input.pric())}
            df = pd.DataFrame(data)
            #fig = px.bar(df,x="Categoria",y="Categoria")
#            title="Valores por Categoría",
#            labels={"Valores": "Valor", "Categoría": "Categoría"},
#            text="Valores"  # Mostrar valores encima de las barras
            return px.bar(df,x="Categoria",y="Valores",title="Distribución de presupuesto")

def mejor_compra_carro(presu):
    # Hacemos 2 variables auxiliares para poder decidir cual es la mejor combinación
    mejor_combinacion = None 
    mejor_valor_total = 0
    # Hacemos un iteración sobre todos los carros
    for i , carro in mtcars.iterrows():
        # Tomamos el precio del carro y se lo restamos al presupuesto
        precio_carro = carro['price'] 
        restante = presu - precio_carro
        # Checamos si existe diamentes que están con el precio restante, de otra forma nos vamos al siguiente carro
        diamonds_com = diamante[diamante['price'] <= restante]
        if diamonds_com.empty:
            continue
        # Tomamos el diamante más caro que se pueda comprar el dinero restante y se lo sumamos al valor del carro
        # para poder checar que tan cerca está el presupuesto y guardamos la suma que nos da
        best_diamond = diamonds_com.loc[diamonds_com['price'].idxmax()]
        valor_total = precio_carro + best_diamond['price']
        # Como esto se repite iterativamente para cada carro, aquí elegimos cuál es el que más se acerca al presupuesto
        # Y se va actualizando la suma (valor_total), el carro y el diamante
        if valor_total > mejor_valor_total:
            mejor_valor_total = valor_total
            mejor_combinacion = {
                "Carro": carro.to_dict()
            }
    # Si existe dicha combinación la regresa, de lo contrario regresa un mensaje de que no existe
    return pd.DataFrame(mejor_combinacion).T

def mejor_compra_diamante(presu):
    # Hacemos 2 variables auxiliares para poder decidir cual es la mejor combinación
    mejor_combinacion = None 
    mejor_valor_total = 0

    # Hacemos un iteración sobre todos los carros
    for i , carro in mtcars.iterrows():
        # Tomamos el precio del carro y se lo restamos al presupuesto
        precio_carro = carro['price'] 
        restante = presu - precio_carro
        # Checamos si existe diamentes que están con el precio restante, de otra forma nos vamos al siguiente carro
        diamonds_com = diamante[diamante['price'] <= restante]
        if diamonds_com.empty:
            continue
        # Tomamos el diamante más caro que se pueda comprar el dinero restante y se lo sumamos al valor del carro
        # para poder checar que tan cerca está el presupuesto y guardamos la suma que nos da
        best_diamond = diamonds_com.loc[diamonds_com['price'].idxmax()]
        valor_total = precio_carro + best_diamond['price']
        # Como esto se repite iterativamente para cada carro, aquí elegimos cuál es el que más se acerca al presupuesto
        # Y se va actualizando la suma (valor_total), el carro y el diamante
        if valor_total > mejor_valor_total:
            mejor_valor_total = valor_total
            mejor_combinacion = {
                "Diamante": best_diamond.to_dict()
            }
    # Si existe dicha combinación la regresa, de lo contrario regresa un mensaje de que no existe
    return pd.DataFrame(mejor_combinacion).T


def mejor_compra_combinada(presu):
    # Hacemos 2 variables auxiliares para poder decidir cual es la mejor combinación
    mejor_combinacion = None 
    mejor_valor_total = 0

    # Hacemos un iteración sobre todos los carros
    for i , carro in mtcars.iterrows():
        # Tomamos el precio del carro y se lo restamos al presupuesto
        precio_carro = carro['price'] 
        restante = presu - precio_carro
        # Checamos si existe diamentes que están con el precio restante, de otra forma nos vamos al siguiente carro
        diamonds_com = diamante[diamante['price'] <= restante]
        if diamonds_com.empty:
            continue
        # Tomamos el diamante más caro que se pueda comprar el dinero restante y se lo sumamos al valor del carro
        # para poder checar que tan cerca está el presupuesto y guardamos la suma que nos da
        best_diamond = diamonds_com.loc[diamonds_com['price'].idxmax()]
        valor_total = precio_carro + best_diamond['price']
        # Como esto se repite iterativamente para cada carro, aquí elegimos cuál es el que más se acerca al presupuesto
        # Y se va actualizando la suma (valor_total), el carro y el diamante
        if valor_total > mejor_valor_total:
            mejor_valor_total = valor_total
            mejor_combinacion = {
                "Suma de los precios": carro['price']+best_diamond['price']
            }
    # Si existe dicha combinación la regresa, de lo contrario regresa un mensaje de que no existe
    return mejor_combinacion["Suma de los precios"]

def mejor_compra(presu):
    # Hacemos 2 variables auxiliares para poder decidir cual es la mejor combinación
    mejor_combinacion = None 
    mejor_valor_total = 0

    # Hacemos un iteración sobre todos los carros
    for i , carro in mtcars.iterrows():
        # Tomamos el precio del carro y se lo restamos al presupuesto
        precio_carro = carro['price'] 
        restante = presu - precio_carro
        # Checamos si existe diamentes que están con el precio restante, de otra forma nos vamos al siguiente carro
        diamonds_com = diamante[diamante['price'] <= restante]
        if diamonds_com.empty:
            continue
        # Tomamos el diamante más caro que se pueda comprar el dinero restante y se lo sumamos al valor del carro
        # para poder checar que tan cerca está el presupuesto y guardamos la suma que nos da
        best_diamond = diamonds_com.loc[diamonds_com['price'].idxmax()]
        valor_total = precio_carro + best_diamond['price']
        # Como esto se repite iterativamente para cada carro, aquí elegimos cuál es el que más se acerca al presupuesto
        # Y se va actualizando la suma (valor_total), el carro y el diamante
        if valor_total > mejor_valor_total:
            mejor_valor_total = valor_total
            mejor_combinacion = [carro['price'],best_diamond['price']]
    # Si existe dicha combinación la regresa, de lo contrario regresa un mensaje de que no existe
    return mejor_combinacion