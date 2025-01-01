from shiny.express import input, render, ui
from shiny import reactive, req
from shinywidgets import render_plotly
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px
from sklearn.mixture import GaussianMixture
from sklearn.cluster import SpectralClustering,AgglomerativeClustering,Birch,KMeans

mtcars = pd.read_csv('mtcars.csv')
var_num = mtcars[['mpg', 'disp', 'hp', 'drat', 'wt', 'qsec']]
data_std = StandardScaler().fit_transform(var_num)
pca = PCA()
pca_results = pca.fit_transform(data_std)
pca_results= pd.DataFrame(pca_results,columns=[f"PC{i+1}" for i in range(6)])


ui.page_opts(title="Principal Components Analysis", fillable=True)

with ui.sidebar():
    ui.input_selectize(
        "var1", "Selecciona componente",
        ["PC1", "PC2", "PC3", "PC4", "PC5", "PC6"]
    )
    ui.input_selectize(
        "var2", "Selecciona componente",
        ["PC1", "PC2", "PC3", "PC4", "PC5", "PC6"]
    )
    
    ui.input_selectize(
        "modelo", "Selecciona el modelo",
        ["K-Means", "Gaussian Mixture Models", "Spectral Clustering", "Birch",'Agglomerative Clustering']
    )
    ui.input_numeric("clt", "Numeros de clusters", 3)
    
with ui.layout_columns(col_widths=[6, 6, 12]):
    with ui.card(full_screen=True):
        ui.card_header("Clusterizacion")
        @render_plotly
        def plot():
            if input.modelo()=='K-Means':
                kmeans = KMeans(n_clusters=input.clt(), random_state=42)
                clusters = kmeans.fit_predict(data_std)
            elif input.modelo()=='Gaussian Mixture Models':
                gmm = GaussianMixture(n_components=input.clt(), random_state=42)
                clusters = gmm.fit_predict(data_std)
            elif input.modelo()=='Spectral Clustering':
                spectral = SpectralClustering(n_clusters=input.clt(), affinity='nearest_neighbors', random_state=42)
                clusters = spectral.fit_predict(data_std)
            elif input.modelo()=='Birch':
                birch = Birch(n_clusters=input.clt())
                clusters = birch.fit_predict(data_std)
            elif input.modelo()=='Agglomerative Clustering':
                agg_clustering = AgglomerativeClustering(n_clusters=input.clt())
                clusters = agg_clustering.fit_predict(data_std)
                
            
            # Añadir los clusters al DataFrame original
            pca_results['Cluster'] = clusters
            return px.scatter(pca_results, x="PC1", y="PC2",
                              symbol="Cluster",# Columna para definir los colores
                              title="Cluterizacion utilizando "+input.modelo()+' con '+str(input.clt())+' clusters')
        
    with ui.card(full_screen=True):
        ui.card_header("Componentes principales")
        @render_plotly
        def plot2():
            return px.scatter(pca_results,x=input.var1(),y=input.var2(),
                              title='Componentes principanes '+ input.var1()+' y '+input.var2())
            #return px.histogram(load_penguins(), x=input.var(), nbins=input.bins())
            
    with ui.card(full_screen=True):
        ui.card_header("Tabla dinamica")
        @render.data_frame
        def table():
            return pca_res()    

def pca_res():
    if input.modelo()=='K-Means':
        kmeans = KMeans(n_clusters=input.clt(), random_state=42)
        clusters = kmeans.fit_predict(data_std)
    elif input.modelo()=='Gaussian Mixture Models':
        gmm = GaussianMixture(n_components=input.clt(), random_state=42)
        clusters = gmm.fit_predict(data_std)
    elif input.modelo()=='Spectral Clustering':
        spectral = SpectralClustering(n_clusters=input.clt(), affinity='nearest_neighbors', random_state=42)
        clusters = spectral.fit_predict(data_std)
    elif input.modelo()=='Birch':
        birch = Birch(n_clusters=input.clt())
        clusters = birch.fit_predict(data_std)
    elif input.modelo()=='Agglomerative Clustering':
        agg_clustering = AgglomerativeClustering(n_clusters=input.clt())
        clusters = agg_clustering.fit_predict(data_std)
    var_num['Cluster'] = clusters     
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
    cluster_summary
    return cluster_summary