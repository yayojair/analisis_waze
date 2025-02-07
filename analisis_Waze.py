#Importaciones de bibliotecas a utilziar
import pandas as pd # manipulacion de datos
import numpy as np # calculos y numeros aleatorios
from scipy.stats import skew # calcula la simetria
import seaborn as sns # Visualización de datos.
import matplotlib.pyplot as plt # Creación de gráficos y visualizaciones.
from scipy.stats import norm # visualizar la distribución en forma de curva
from math import sqrt # Obtener raiz cuadras
import os #abrir directorios
import random #numeros aleatorios
from matplotlib.backends.backend_pdf import PdfPages # guarda graficos en un pdf
from multiprocessing import Pool, cpu_count #concurrente
import math # operaciones matematicas
from geopy.distance import geodesic #distancias de havernise
import geopandas as gpd #abrir archivos geojson
from shapely.geometry import Point, Polygon #clases de Shapely que permiten representar puntos y áreas geográficas.
import osmnx as ox #biblioteca que facilita la obtención y el análisis de datos geoespaciales de OpenStreetMap.
from dotenv import dotenv_values # lectura de archivo .env


def diagrama_dispersion(df_datos, df_cdmx,coord_x, coord_y, pdf):
    """
    Grafica las coordenadas de los perimetros y de las alertas de la cdmx, se visualiza las divisiones (casillas)
    y guarda dicha grafica en un archivo pdf.


    Args:
        df_datos (DataFrame): base de datos que contienen las alertas.
        df_cdmx (DataFrame): base de datos que contienen las coordenadas del poligono de la cdmx.
        coord_x (list): coordenadas de las divisiones en el eje x.
        coord_y (list): coordenadas de las divisiones en el eje y.
        pdf (PdfPages): archivo donde se guarda la grafica.


    Returns:
        None
   
    Raises:
        None
    """
    # Crear una figura con un tamaño de 20x16 pulgadas
    plt.figure(figsize=(20, 16))
   
    # primer conjunto de datos se dibuja en rojo
    sns.scatterplot(x='longitud', y='latitud', data=df_datos, color='blue', edgecolor='blue',  s=2)


    # segundo conjunto de datos se dibuja en rojo
    sns.scatterplot(x='longitud', y='latitud', data=df_cdmx, color='red', edgecolor='red',  s=2)


    # dividir el diagrama (casillas)
    plt.xticks(coord_x)
    plt.yticks(coord_y)


    # Mostrar una cuadrícula de fondo para facilitar la visualización del gráfico
    plt.grid(which='both', color='gray', linestyle='--', linewidth=0.5)


    # Añadir etiquetas a los ejes x e y para describir qué representan
    plt.xlabel('Longitud')
    plt.ylabel('Latitud')


    # Mostrar una cuadrícula adicional (esta vez se activa completamente con el parámetro True)
    plt.grid(True)
    pdf.savefig()
    plt.close()


def histograma_curva(datos, columna, activar_frec,pdf):
    """
    Grafica el comportamiento de los datos de la variable columna mediante un histograma y muestra su curva de densidad
    para verificar si se tiene datos atipicos y guarda dicha grafica en un archivo pdf.


    Args:
        df_datos (DataFrame): base de datos que contienen las alertas.
        columna (string): nombre de la columna que contiene los valores numericos de la base de datos df_datos.
        activar_frec (bool): activa la curva de densidad (True) o la frecuancia (False).
        pdf (PdfPages): archivo donde se guarda la grafica.


    Returns:
        None
   
    Raises:
        None
    """
   
    datos_visualizar = datos[columna]


    # Crear la figura y el histograma
    plt.figure(figsize=(10, 6))
    plt.hist(datos_visualizar, bins=20, color='skyblue', edgecolor='black', density=activar_frec)  # Ajustar histograma para densidad


    # Calcular la media y la desviación estándar de los datos
    media_datos = datos_visualizar.mean()
    desviacion_estandar = datos_visualizar.std()
    resumen = datos_visualizar.describe()


    # Obtener los límites del eje x
    xmin, xmax = plt.xlim()


    # Crear un rango de valores en el eje x
    x = np.linspace(xmin, xmax, 100)


    # Obtener la función de densidad de probabilidad
    probabilidad = norm.pdf(x, media_datos, desviacion_estandar)


    # Añadir la curva de densidad al gráfico
    plt.plot(x, probabilidad, 'k', linewidth=2)  # 'k' es el color negro


    # Añadir líneas para la media, los cuartiles
    plt.axvline(datos_visualizar.quantile(0.25), color='orange', linestyle='dashed', linewidth=1.5)
    plt.axvline(datos_visualizar.quantile(0.75), color='orange', linestyle='dashed', linewidth=1.5)
    plt.axvline(datos_visualizar.quantile(0.50), color='pink', linestyle='dashed', linewidth=1.5)
    plt.axvline(media_datos, color='black', linestyle='dashed', linewidth=1.5)


    # Añadir etiquetas y título
    plt.xlabel(columna)
    if activar_frec == True:
        y_name = 'Densidad'
    else:
        y_name = 'Frecuencia'
    plt.ylabel(y_name)
    plt.title('Histograma y Curva de ' + y_name +' de '+ columna)


    # Mostrar la leyenda
    resumen = f'Resumen: \n{resumen}'
    plt.text(0.95, 0.95, resumen,
             horizontalalignment='right', verticalalignment='top',
             transform=plt.gca().transAxes, fontsize=10)


    # Mostrar la cuadrícula
    plt.grid(True)
    pdf.savefig()
    plt.close()


def caja_bigote(datos, columna, pdf):
    """
    Grafica el comportamiento de los datos mediantes sus cuartiles, mostrando los datos atipicos
    que se presentan y guarda dicha grafica en un archivo pdf.


    Args:
        df_datos (DataFrame): base de datos que contienen las alertas.
        columna (string): nombre de la columna que contiene los valores numericos de la base de datos df_datos.
        pdf (PdfPages): archivo donde se guarda la grafica.


    Returns:
        None
   
    Raises:
        None
    """


    # Se configura el tamaño de la figura para el gráfico (10 x 6 pulgadas)
    plt.figure(figsize=(10, 6))
   
    # Se utiliza la función boxplot de Seaborn para crear el gráfico de caja
    # El parámetro 'y' indica que la variable a graficar está en la columna 'columna'
    # 'palette' define la paleta de colores a usar para el gráfico (en este caso 'Set2')
    sns.boxplot(y=columna, data=datos, palette='Set2')


    # Añadir etiquetas descriptivas a los ejes
    plt.ylabel(columna)  
    plt.title(f'Diagrama de Caja y Bigotes de {columna}')


    pdf.savefig()
    plt.close()


def agregar_puntos(particiones, divisiones_x, coord_x, coord_y):
    """
    Se agrega un punto al centro da cada cuadricula donde cada proceso empezara desde una fila en particular.


    args:
        particiones (list): valores donde le indica a cada proceso donde empezar y donde terminar su recorrido.
        divsiones_x (int): numero de divisiones en x.
        coord_x (list): coordenas de las divisiones en x.
        coord_y (list): coordenas de las divisiones en y.
   
    return:
        almacen_coord (list): coordenadas de los puntos que fueron agregados.


    Raises:
        None


    """
    inicia = particiones[0]
    termina = np.max(particiones)+1
    almacen_coord = []
    for y in range(inicia, termina): # itera por el eje
        for x in range(divisiones_x): # itera por el eje x
            lon = (coord_x[x] + coord_x[x+1]) / 2
            lat = (coord_y[y] + coord_y[y+1]) / 2
            almacen_coord.append(Point(lon, lat))
    return almacen_coord


def contar_alertas(particiones, divisiones_x, divisiones_y, coord_x, coord_y, datos_cdmx):
    """
    Cada proceso recorre sus respectivas filas para asignar un numero de casilla y contar cuantas alertas hay en cada casilla  


    args:
        particiones (list): valores donde le indica a cada proceso donde empezar y donde terminar su recorrido.
        divisiones_x (int): numero de divisiones en x.
        divisiones_y (int): numero de divisiones en y.
        coord_x (list): coordenas de las divisiones en x.
        coord_y (list): coordenas de las divisiones en y.
        datos_cdmx (DataFrame): base de datos donde se contiene las alertas.
   
    return:
        tuplas:casilla_datos (tuple): indices de las alertas, numero de la casilla y cantidad de alertas  
               datos_analizar (tuple): numero de casillas y alertas


    Raises:
        None


    """
    inicia = particiones[0]
    termina = np.max(particiones)+ 1
    casillas = (inicia * divisiones_x) + 1
    casilla_datos = []
    datos_analizar = []
    for y in range(inicia, termina): # itera por el eje
        for x in range(divisiones_x): # itera por el eje x
            if (x+1 == divisiones_x) & (y+1 == divisiones_y):
                df_aux_datos = datos_cdmx[(datos_cdmx['latitud'] >= coord_y[y]) & (datos_cdmx['latitud'] <= coord_y[y+1]) & (datos_cdmx['longitud'] >= coord_x[x]) & (datos_cdmx['longitud'] <= coord_x[x+1])]
            elif y+1 == divisiones_y:
                df_aux_datos = datos_cdmx[(datos_cdmx['latitud'] >= coord_y[y]) & (datos_cdmx['latitud'] <= coord_y[y+1]) & (datos_cdmx['longitud'] >= coord_x[x]) & (datos_cdmx['longitud'] < coord_x[x+1])]
            elif x+1 == divisiones_x:
                df_aux_datos = datos_cdmx[(datos_cdmx['latitud'] >= coord_y[y]) & (datos_cdmx['latitud'] < coord_y[y+1]) & (datos_cdmx['longitud'] >= coord_x[x]) & (datos_cdmx['longitud'] <= coord_x[x+1])]
            else:
                df_aux_datos = datos_cdmx[(datos_cdmx['latitud'] >= coord_y[y]) & (datos_cdmx['latitud'] < coord_y[y+1]) & (datos_cdmx['longitud'] >= coord_x[x]) & (datos_cdmx['longitud'] < coord_x[x+1])]
            alertas = df_aux_datos.shape[0]
            if alertas > 0:
                casilla_datos.append([df_aux_datos.index, casillas, alertas])
                if alertas > 1:
                    datos_analizar.append([casillas, (alertas-1)])
            casillas += 1
    return casilla_datos, datos_analizar


def valores_atipicos(datos, columna, superior, inferior):
    """
    Se encuentra los indices del data frame que se consideran atipicos, utilizando el rango intercuartil.


    args:
        datos (DataFrame): base de datos que contiene los datos a analizar como casilla y cantidad de alertas.
        columna (string): nombre de la columna que contiene los valores numericos de la base de datos df_datos.
        superior (list): almacen para guardar los indices superior de los datos atipicos.
        inferior (list): almacen para guardar los indices inferiores de los datos atipicos.


    return:
        None




    """
    # El primer cuartil (Q1) es el valor en el 25% de los datos, el tercer cuartil (Q3) es el valor en el 75%
    primer_cuartil = datos[columna].quantile(0.25)
    tercer_cuartil = datos[columna].quantile(0.75)


    # Calcular el rango intercuartílico (IQR), que es la diferencia entre el tercer cuartil y el primer cuartil
    iqr = tercer_cuartil - primer_cuartil


    # Definir los valores límite para los valores atípicos:
    # Cualquier valor mayor que (Q3 + 1.5*IQR) es considerado atípico superior
    # Cualquier valor menor que (Q1 - 1.5*IQR) es considerado atípico inferior
    valores_atipico_mayor = tercer_cuartil + 1.5 * iqr
    valores_atipico_menor = primer_cuartil - 1.5 * iqr


    # Encontrar los índices de los valores atípicos en la parte superior (mayores que el valor límite superior)
    indice_superior = datos[datos[columna] > valores_atipico_mayor].index
   
    # Encontrar los índices de los valores atípicos en la parte inferior (menores que el valor límite inferior)
    indice_inferior = datos[datos[columna] < valores_atipico_menor].index


    # Añadir los índices de los valores atípicos superiores e inferiores a las listas proporcionadas
    superior.append(indice_superior)
    inferior.append(indice_inferior)


def perimetro():
    """
    Se geocodificar una ubicación (cdmx) con el fin de extraer las coordenadas que forma el poligono para
    calcular mediante la medida de 4 cuadras cuantas divisiones (cuadricula) se tiene que hacer en la CDMX,
    se encuentra las coordedas de dichas divisiones.


    Args:
        none


    Returns:
        tupla: list: coordenadas de las divisiones en x.
               list: coordenadas de las divisiones en y.
               int: numero de divisiones en x.
               int: numero de divisiones en y.
               Polygon: poligono de la cdmx
    Raises:
        None
    """
    #obtener poligono de la cdmx
    # ox.geocode_to_gdf función que permite geocodificar una ubicación
    # devuelve un GeoDataFrame de geopandas con los resultados de la geocodificación.
    cdmx = ox.geocode_to_gdf("Ciudad de México, México")
    cdmx_coord = cdmx.geometry.iloc[0].exterior.coords
    poligono = Polygon(cdmx_coord)
    df_cdmx = pd.DataFrame(data=cdmx_coord, columns=['longitud', 'latitud'])


    #divisiones para la cuadricula


        #distancias de las cuadras
    distancia_y_cuadras = geodesic((19.40213, -99.14199), (19.39926, -99.14249)).meters
    distancia_x_cuadras = geodesic((19.39926, -99.14249), (19.39901, -99.14077)).meters


        # coordenas del punto maximo, minimo en el eje x, y del perimetro de la cdmx
    punto_x_min = tuple(df_cdmx.iloc[np.argmin(df_cdmx['longitud'])][['latitud', 'longitud']])
    punto_x_max = tuple(df_cdmx.iloc[np.argmax(df_cdmx['longitud'])][['latitud','longitud']])
    punto_y_min = tuple(df_cdmx.iloc[np.argmin(df_cdmx['latitud'])][['latitud','longitud']])
    punto_y_max = tuple(df_cdmx.iloc[np.argmax(df_cdmx['latitud'])][['latitud','longitud']])


        # se pone en los mismos ejes para obtener la distancias
    punto_x_min = (punto_x_max[0],punto_x_min[1])
    punto_y_min = (punto_y_min[0], punto_y_max[1])


        # se obtiene la distancia utilizando la distancia de haversine, para saber cuantas casillas tendra el mapa
    distancia_x_perimetro = geodesic(tuple(punto_x_max), tuple(punto_x_min)).meters
    distancia_y_perimetro = geodesic(tuple(punto_y_max), tuple(punto_y_min)).meters  


        #se calcula la cantidad de casillas que se va a tener en el eje xy
    divisiones_x = round( distancia_x_perimetro  / distancia_x_cuadras)
    divisiones_y = round( distancia_y_perimetro / distancia_y_cuadras)


        # numero de divisiones
    print(divisiones_x)
    print(divisiones_y)
   
        # Crear los límites de las casillas
    coord_y = np.linspace(punto_y_min[0], punto_y_max[0], divisiones_y+1)
    coord_x = np.linspace(punto_x_min[1], punto_x_max[1], divisiones_x+1)      
    return coord_x, coord_y, divisiones_x, divisiones_y, df_cdmx, poligono


def tareas_procesos(divisiones_y, divisiones_x, coord_x, coord_y, datos_cdmx):
    """
    La función se encarga de dividir el trabajo de análisis en múltiples procesos para acelerar la ejecución 
    mediante procesamiento paralelo. Se encarga de repartir la tarea de agregar puntos
    o contar alertas en un mapa, dependiendo de si hay datos disponibles en datos_cdmx.
    
    Args:
        args:
        particiones (list): valores donde le indica a cada proceso donde empezar y donde terminar su recorrido.
        divisiones_x (int): numero de divisiones en x.
        divisiones_y (int): numero de divisiones en y.
        coord_x (list): coordenas de las divisiones en x.
        coord_y (list): coordenas de las divisiones en y.
        datos_cdmx (DataFrame): base de datos donde se contiene las alertas.

    Returns:
        datos_procesos (Object): regresa los valores obtenidos por los procesos.

    Raises:
        None
    """
    #agregar puntos a todo el mapa repartiendo la tarea a procesos
    numero_proceso = min(divisiones_y, cpu_count())
    valores_proceso = [valor for valor in range(divisiones_y)]
    #divide las filas que le toca a cada proceso
    particiones_valores = np.array_split(valores_proceso, numero_proceso)
    with Pool(processes=numero_proceso) as pool:
        if datos_cdmx is None:
            args = [(particiones, divisiones_x, coord_x, coord_y)for particiones in particiones_valores]
            #starmap funcion que sirve para pasar mas de un parametro a las tarea de los procesos
            datos_procesos = pool.starmap(agregar_puntos, args)
        else:
            args = [(particiones, divisiones_x, divisiones_y, coord_x, coord_y, datos_cdmx)for particiones in particiones_valores]
            datos_procesos = pool.starmap(contar_alertas, args)
    
    return datos_procesos

def clasificar_datos(media, std):
    """
    La función se encarga de clasificar los datos de datos_cdmx en niveles de alertas.
    
    Args:
       media (int): media de los datos.
       std (int): desviacion estandar de los datos
        

    Returns:
        datos_cdmx (data frame): datos con la clasificacion de alertas.

    Raises:
        None
    """
    if media == std:
        std = std - 1 
    print(f'promedio: {media}')
    print(f'std: {std}')
    alto = media+std+2
    print(f'alto {alto}')
    mod_alta = media+1
    print(f'mod_alta {mod_alta}')
    mod_baja = abs(media-std)+1
    print(f'{mod_baja}')
    datos_cdmx['nivel_alertas'] = datos_cdmx['cant_alertas'].apply(lambda x: 'alto' if x >  alto else
                                                                'modo_alta' if (x > mod_alta) & (x<=alto) else
                                                                'modo_bajo' if (x > mod_baja) & (x <=mod_alta) else
                                                                'bajo' if (x>1) & (x <= mod_baja) else
                                                                'nulo')
    return datos_cdmx       


if __name__ == '__main__':
    #abrir base de datos
    ruta_waze = '/home/local/recoleccion_datos/datos_2019.csv'
    df_datos = pd.read_csv(ruta_waze)

    #eliminar datos duplicados
    df_datos.drop_duplicates(subset=['id'], inplace=True) #borrar datos duplicados que tenga el mismo id
    df_datos.drop_duplicates(subset=['longitud', 'latitud'], inplace=True)
    df_datos = df_datos.reset_index(drop=True) #reinicia los indices sin crear columna de los indices anteriores
    print(df_datos.info())

    #informacion respecto al poligono y cuadricula 
    datos = perimetro()
    coord_x = datos[0]
    coord_y = datos[1] 
    divisiones_x = datos[2]
    divisiones_y = datos[3] 
    df_cdmx = datos[4]
    poligono = datos[5]

    #recolectar los puntos agregados de los proceso
    puntos_agregados = tareas_procesos(divisiones_y, divisiones_x, coord_x, coord_y, None)
    puntos_faltantes = []
    for i in range(len(puntos_agregados)):
        puntos_faltantes = puntos_faltantes + puntos_agregados[i]
    

    alertas_coord = []
    #convertir la columna longitud y latitud en puntos geometricos
    alertas_coord = df_datos.apply(lambda x: Point(x['longitud'], x['latitud']), axis=1).tolist()

    #juntar los puntos geometricos 
    alertas_coord = puntos_faltantes + alertas_coord
    #eliminar los puntos que no pertenecen a la cdmx
    cdmx_coord = [(punto.x, punto.y) for punto in alertas_coord if poligono.contains(punto)]

    #data frame de los datos que pertenecen a la cdmx
    datos_cdmx = pd.DataFrame(data=cdmx_coord, columns=['longitud', 'latitud'])

    nombre_pdf = 'Visualizar_2019_prueba.pdf' #nombre del pdf
    with PdfPages(nombre_pdf) as pdf:
        diagrama_dispersion(datos_cdmx, df_cdmx,coord_x, coord_y, pdf)
        #se crea dos columnas en el dataframe de lso datos 
        datos_cdmx['casilla'] = pd.NA 
        datos_cdmx['cant_alertas'] = pd.NA
        
        tuplas_datos = tareas_procesos(divisiones_y, divisiones_x, coord_x, coord_y, datos_cdmx)
        #agregar las casillas y la cantidad de alerta a las alertas 
        datos_analizar = []
        for i in range(len(tuplas_datos)):
            datos = tuplas_datos[i]
            datos_analizar = datos_analizar + datos[1]
            for j in range(len(datos[0])):
                alertas = datos[0][j]
                indices = alertas[0]
                datos_cdmx.loc[indices,'casilla'] = alertas[1]
                datos_cdmx.loc[indices,'cant_alertas'] = alertas[2]

        #Data Frame de los datos a analizar 
        datos_analizar = pd.DataFrame(data=datos_analizar, columns=['casilla', 'cant_alertas'])
        caja_bigote(datos_analizar , 'cant_alertas', pdf)
        histograma_curva(datos_analizar , 'cant_alertas', True, pdf)

        #eliminar datos atipicos
        atipicos = True       
        while(atipicos):
            indices_superior = []
            indices_inferior = []
            valores_atipicos(datos_analizar, 'cant_alertas', indices_superior, indices_inferior)
            if (len(indices_superior[0]) != 0):
                datos_analizar = datos_analizar.drop(index=indices_superior[0])  
            else:
                atipicos=False
            datos_analizar = datos_analizar.reset_index(drop=True)

        #visualizar datos sin valores atipicos
        caja_bigote(datos_analizar , 'cant_alertas', pdf)
        histograma_curva(datos_analizar , 'cant_alertas', True, pdf)

    media = round(datos_analizar['cant_alertas'].mean())
    std = round(datos_analizar['cant_alertas'].std())

    datos_cdmx = clasificar_datos(media, std)
    datos_cdmx.to_csv('datos_finales_2019.csv')
    print(datos_cdmx['nivel_alertas'].unique())
    
    print('termino')
