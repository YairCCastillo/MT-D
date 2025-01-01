import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from itertools import accumulate
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

mtcars = pd.read_csv('mtcars.csv')
diamonds = pd.read_csv('diamonds.csv')

def PCA_mtcars():
    # Leer el archivo
    mtcars = pd.read_csv('mtcars.csv')
    
    # Se toman solo las variables numéricas
    var_num = mtcars[['mpg', 'disp', 'hp', 'drat', 'wt', 'qsec']]
    
    # Estandarizamos los datos, debido a que tiene rangos muy diferentes
    data_std = StandardScaler().fit_transform(var_num)
    
    # Hacemos PCA 
    pca = PCA()
    pca_results = pca.fit_transform(data_std)
    pca_results = pd.DataFrame(pca_results,columns=[f"PC{i+1}" for i in range(6)])
    print('ANALISIS CON PCA')
    # Obtemos la varianza explicada por los componentes y graficamos
    expl_var = pca.explained_variance_ratio_
    var_exp = list(accumulate(expl_var))
    plt.figure(figsize=(8, 5))
    plt.plot(range(1,len(var_exp)+1), var_exp, marker='o', linestyle='-', linewidth=2)
    plt.title("Varianza Explicada acumulada")
    plt.xlabel("Componente Principal")
    plt.ylabel("Varianza")
    plt.show()
    loadings = pca.components_.T
    print('BIPLOT')
    # Biplot
    plt.figure(figsize=(12, 8))
    # Scatter plot de las puntuaciones de los componentes principales
    plt.scatter(pca_results['PC1'], pca_results['PC2'], alpha=0.7, label='Carros')
    # Les ponemos etiquestas
    for i, label in enumerate(mtcars.index):
        plt.text(pca_results['PC1'][i], pca_results['PC2'][i], label, fontsize=9, alpha=0.7)
    # Graficamos las lineas    
    for i, var in enumerate(var_num.columns):
        plt.arrow(0, 0, loadings[i, 0] * max(pca_results['PC1']), loadings[i, 1] * max(pca_results['PC2']),
                  color='r', alpha=0.8, head_width=0.05)
        plt.text(loadings[i, 0] * max(pca_results['PC1']) * 1.1,
                 loadings[i, 1] * max(pca_results['PC2']) * 1.1,
                 var, color='r', ha='center', va='center')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Biplot PCA')
    plt.grid()
    plt.legend()
    plt.show()
    print('CLUSTERIZACION')
    # Aplicar K-Means con 3 clusters
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(data_std)
    
    # Añadir los clusters al DataFrame original
    var_num['Cluster'] = clusters
    
    
    # Visualizar los resultados en el espacio PCA
    plt.figure(figsize=(12, 8))
    for i, label in enumerate(var_num.index):
        plt.text(pca_results['PC1'][i], pca_results['PC2'][i], label, fontsize=9, alpha=0.7)
    for cluster in range(3):
        cluster_points = pca_results[var_num['Cluster'] == cluster]
        plt.scatter(cluster_points['PC1'], cluster_points['PC2'], label=f'Cluster {cluster}', alpha=0.7)
    centroids_pca = pca.transform(kmeans.cluster_centers_)
    plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], c='black', s=200, marker='X', label='Centroides')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('K-Means Clustering en Espacio PCA')
    plt.legend()
    plt.grid()
    plt.show()

    # Un resumen de las variables, tomé la media y la desviación estandar
    cluster_summary = var_num.groupby('Cluster').agg({
    'mpg': ['mean', 'std'],
    'disp': ['mean', 'std'],
    'hp': ['mean', 'std'],
    'wt': ['mean', 'std'],
    'qsec': ['mean', 'std'],
    'drat': ['mean', 'std']
    })
    # Renombrar columnas para mayor claridad
    cluster_summary.columns = ['_'.join(col).strip() for col in cluster_summary.columns]
    cluster_summary.reset_index(inplace=True)
    print(cluster_summary)

def prob_cost_may(precio):
    total_diamantes = len(diamonds)
    check_mayor = len(diamonds[diamonds['price'] > precio])
    return check_mayor / total_diamantes
def simulacion_compra_sep(presu):
    mtcars = pd.read_csv('mtcars.csv')
    diamonds = pd.read_csv('diamonds.csv')
    # ASUMIMOS QUE MPG MULTIPLICADO POR MIL ES EL PRECIO DEL CARRO
    mtcars['price'] = mtcars['mpg']*1000
    # Vemos que carros y diamantes están dentro del prespuesto
    carros_com = mtcars[mtcars['price'] <= presu]
    diamonds_com = diamonds[diamonds['price'] <= presu]

    # Checamos si están vacíos los resultados y regresamos un mensaje de ser verdad
    if diamonds_com.empty and carros_com.empty:
        return "No hay carros ni diamentes dentro de ese presupuesto"

    # Puede que para carros no alcance pero para diamantes sí
    if carros_com.empty:
        print('No hay carro de ese presupuesto pero diamente sí:')
        # Se elige al azar el diamante
        mejor_diam = diamonds_com.loc[np.random.choice(diamonds_com.index)]
        return {"Mejor diamante": mejor_diam.to_dict()}
    # Puede que para diamantes no alcance pero para carro sí
    elif diamonds_com.empty:
        print("No hay diamentes dentro de ese presupuesto pero carro sí:")
        # Se elige al azar el carro
        mejor_carro = carros_com.iloc[np.random.choice(carros_com.index)]
        return {"Mejor carro": mejor_carro.to_dict()}

    # Si todo lo anterior no se cumple, entonces sí existe carro y diamante con ese prespuesto
    mejor_diam = diamonds_com.loc[np.random.choice(diamonds_com.index)]
    mejor_carro = carros_com.loc[np.random.choice(carros_com.index)]
    # Mostramos los resultados
    return {
        "Mejor carro": mejor_carro.to_dict(),
        "Mejor diamante": mejor_diam.to_dict()
    }
def simulacion_compra_junto(presu):
    mtcars = pd.read_csv('mtcars.csv')
    diamonds = pd.read_csv('diamonds.csv')
    # ASUMIMOS QUE MPG MULTIPLICADO POR MIL ES EL PRECIO DEL CARRO
    mtcars['price'] = mtcars['mpg']*1000
    # Vemos que carros y diamantes están dentro del prespuesto
    carros_com = mtcars[mtcars['price'] <= presu]
    diamonds_com = diamonds[diamonds['price'] <= presu]

    # Checamos si están vacíos los resultados y regresamos un mensaje de ser verdad
    if diamonds_com.empty and carros_com.empty:
        return "No hay carros ni diamentes dentro de ese presupuesto"
    if carros_com.empty:
        print('No hay carro de ese presupuesto pero diamente sí:')
        # Se elige al azar el diamante
        mejor_diam = diamonds_com.loc[np.random.choice(diamonds_com.index)]
        return {"Mejor diamante": mejor_diam.to_dict()}
    elif diamonds_com.empty:
        print("No hay diamentes dentro de ese presupuesto pero carro sí:")
        # Se elige al azar el carro
        mejor_carro = carros_com.loc[np.random.choice(carros_com.index)]
        return {"Mejor carro": mejor_carro.to_dict()}

    # Ya teniendo el carro, sacamos el dinero restante
    mejor_carro = carros_com.loc[np.random.choice(carros_com.index)]
    restante = presu - mejor_carro['price']
    # Checamos diamantes que estén debajo de ese restante
    diamonds_com = diamonds[diamonds['price'] <= restante]
    # Si no hay solo regresamos el carro
    if diamonds_com.empty:
        print("No hay diamentes dentro de ese presupuesto restante")
        return {"Mejor carro": mejor_carro.to_dict()}
    # Elegimos el diamante al azar    
    mejor_diam = diamonds_com.loc[np.random.choice(diamonds_com.index)]
    # Mostramos los resultados
    return {
        "Mejor carro": mejor_carro.to_dict(),
        "Mejor diamante": mejor_diam.to_dict(),
        "Suma de los precios": mejor_carro['price']+mejor_diam['price']
    }
def mejor_compra_combinada(presu):
    mtcars = pd.read_csv('mtcars.csv')
    diamonds = pd.read_csv('diamonds.csv')
    # ASUMIMOS QUE MPG MULTIPLICADO POR MIL ES EL PRECIO DEL CARRO
    mtcars['price'] = mtcars['mpg']*1000
    # Hacemos 2 variables auxiliares para poder decidir cual es la mejor combinación
    mejor_combinacion = None 
    mejor_valor_total = 0

    # Hacemos un iteración sobre todos los carros
    for i , carro in mtcars.iterrows():
        # Tomamos el precio del carro y se lo restamos al presupuesto
        precio_carro = carro['price'] 
        restante = presu - precio_carro
        # Checamos si existe diamentes que están con el precio restante, de otra forma nos vamos al siguiente carro
        diamonds_com = diamonds[diamonds['price'] <= restante]
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
                "Carro": carro.to_dict(),
                "Diamante": best_diamond.to_dict(),
                "Suma de los precios": carro['price']+best_diamond['price']
            }
    # Si existe dicha combinación la regresa, de lo contrario regresa un mensaje de que no existe
    if mejor_combinacion:
        return mejor_combinacion
    else:
        return "No hay posible combinación de diamantes y carros con el presupuesto dado"

def diamantes_pro():
    # CARGAMOS LOS DATOS PARA LA TAREA
    mtcars = pd.read_csv('mtcars.csv')
    diamonds = pd.read_csv('diamonds.csv')
    # ASUMIMOS QUE MPG MULTIPLICADO POR MIL ES EL PRECIO DEL CARRO
    mtcars['price'] = mtcars['mpg']*1000
    # Como queremos el precio por quilate, entonces tenemos que hacer la división entre price por carat
    diamonds['pre_x_qui'] = diamonds['price'] / diamonds['carat']
    # Agrupamos por corte y color y tomamos el promedio
    diam_precio_x_qui = diamonds.groupby(['cut', 'color'])['pre_x_qui'].mean().reset_index()
    print(diam_precio_x_qui)
    print('---------')
    print("Ejemplo de elegir un precio y la probabilidad de que un diamante cueste más que ese precio")
    print("Ejemplo 3000 da una probabilidad de: ",prob_cost_may(3000))
    print("Ejemplo 18000 da una probabilidad de: ",prob_cost_may(18000))
    
def combinacion_pro():
    print("Para este inciso hice 2 funciones, una que da el al azar carro y el diamante menor al presupuesto dado por separado y otro donde los da junto")
    print('Ejemplo, si se tiene un presupuesto  de 20000, estos son los resultados para el separada')
    print(pd.DataFrame.from_dict({list(simulacion_compra_sep(20000).keys())[0]: simulacion_compra_sep(20000)[list(simulacion_compra_sep(20000).keys())[0]]}, orient='index'))
    print(pd.DataFrame.from_dict({list(simulacion_compra_sep(20000).keys())[1]: simulacion_compra_sep(20000)[list(simulacion_compra_sep(20000).keys())[1]]}, orient='index'))
    print('--------------')
    print('Ejemplo, si se tiene un presupuesto  de 20000, estos son los resultados para el junto, ya que la suma tiene que ser menor al presupuesto')
    print(pd.DataFrame.from_dict({list(simulacion_compra_junto(20000).keys())[0]: simulacion_compra_junto(20000)[list(simulacion_compra_junto(20000).keys())[0]]}, orient='index'))
    print(pd.DataFrame.from_dict({list(simulacion_compra_junto(20000).keys())[1]: simulacion_compra_junto(20000)[list(simulacion_compra_junto(20000).keys())[1]]}, orient='index'))
    print('-------------')
    print("Para el último inciso se optimizó para aprovechar todo el presupuesto, en este caso se utilizó también 20000")
    print(pd.DataFrame.from_dict({list(mejor_compra_combinada(20000).keys())[0]: mejor_compra_combinada(20000)[list(mejor_compra_combinada(20000).keys())[0]]}, orient='index'))
    print(pd.DataFrame.from_dict({list(mejor_compra_combinada(20000).keys())[1]: mejor_compra_combinada(20000)[list(mejor_compra_combinada(20000).keys())[1]]}, orient='index'))

def pronostico():
    mtcars = pd.read_csv('mtcars.csv') # Cargamos los archivos
    # Lo hacemos serie de tiempo para un mejor manejo
    mpg_series = pd.Series(mtcars['mpg'].values, 
            index=pd.date_range(start='2023-01-01', periods=len(mtcars), freq='ME'))
    # Graficamos la serie de tiempo
    plt.figure(figsize=(10, 5))
    plt.plot(mpg_series, label="MPG Serie de Tiempo")
    plt.title("(MPG) Series de Tiempo")
    plt.xlabel("Tiempo")
    plt.ylabel("MPG")
    plt.legend()
    plt.show()
    # Hacemos la descoposición
    decomposition = seasonal_decompose(mpg_series, model='additive')
    
    # Graficamos las descomposición
    fig, axes = plt.subplots(4, 1, figsize=(8, 8), sharex=True)
    decomposition.observed.plot(ax=axes[0], title='Observed')
    decomposition.trend.plot(ax=axes[1], title='Trend')
    decomposition.seasonal.plot(ax=axes[2], title='Seasonal')
    decomposition.resid.plot(ax=axes[3], title='Residual')
    plt.tight_layout()
    plt.show()
    print('--------------')
    # Prueba de Dickey-Fuller
    result = adfuller(mtcars['mpg'])
    print('Hacemos pruebas estadísticas para ver si es estacionaria')
    print('Estadístico ADF:', result[0])
    print('p-value:', result[1])
    
    # Si el p-value es mayor a 0.05, la serie no es estacionaria.
    # Diferenciar la serie temporal
    df_diff =mtcars['mpg'].diff().dropna()
    
    # Comprobar si es estacionaria después de la diferencia
    result_diff = adfuller(df_diff)
    print('p-value después de la diferencia:', result_diff[1])
    print('--------------')
    print('Se divide los datos en train y test para poder lograr un buen modelo con ARIMA y se grafica')
    tscv = TimeSeriesSplit(n_splits=4)
    errors = []
    
    for train_index, test_index in tscv.split(mpg_series):
        train, test = mpg_series.iloc[train_index], mpg_series.iloc[test_index]
        model_cv = ARIMA(train, order=(6, 0,2)).fit()
        predictions = model_cv.forecast(steps=len(test))
        rmse = np.sqrt(mean_squared_error(test, predictions))
        errors.append(rmse)
    
    plt.figure(figsize=(10, 5))
    plt.plot(mpg_series, label="Datos")
    plt.plot(predictions, label="Predicciones")
    #plt.plot(pd.concat([predictions,forecast]).index, pd.concat([predictions,forecast]), label="Predicción", linestyle="--", marker='o')
    plt.title("Predicción usando ARIMA para serie de tiempo mpg")
    plt.xlabel("Tiempo")
    plt.ylabel("mpg")
    plt.legend()
    plt.show()
    
    std_rmse = np.std(errors)
    
    print('Errores:',errors)
    print('STD_RMSE:',std_rmse)
    print('--------------')
    print('Por último se toma el mejor modelo que se entrenó y se aplica para predecir los siguiente 6 meses')
    arima_model = ARIMA(mpg_series, order=(6, 0,2))
    arima_fit = arima_model.fit()
    
    # Predicción para los 6 meses siguiente
    forecast = arima_fit.forecast(steps=6)
    
    # Graficar
    plt.figure(figsize=(10, 5))
    plt.plot(mpg_series, label="Datos")
    plt.plot(forecast.index, forecast, label="Predicción", linestyle="--", marker='o')
    plt.title("Predicción usando ARIMA para serie de tiempo mpg")
    plt.xlabel("Tiempo")
    plt.ylabel("mpg")
    plt.legend()
    plt.show()

