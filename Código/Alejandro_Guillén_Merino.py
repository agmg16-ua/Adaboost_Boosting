import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import time as time
import random as random

from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

import time

#Nos permite cargar los datos de entrenamiento y test
"""
(X_train, Y_train), (X_test, Y_test) = keras.datasets.mnist.load_data()
print(X_train.shape, X_train.dtype)
print(Y_train.shape, Y_train.dtype)
print(X_test.shape, X_test.dtype)
print(Y_test.shape, Y_test.dtype)
"""

#Nos permite visualizar las imágenes de los números manuscritos
"""
def show_image(imagen, title):
    plt.figure()
    plt.suptitle(title)
    plt.imshow(imagen, cmap = "Greys")
    plt.show()
    
for i in range(3):
    title = "Mostrando imagen X_train[" + str(i) + "]"
    title = title + " -- Y_train[" + str(i) + "] = " + str(Y_train[i])
    show_image(X_train[i], title)
"""

#Nos permite visualizar los valores de los píxeles de una imagen
"""
def plot_X(X, title, fila, columna):
    plt.title(title)
    plt.plot(X)
    plt.xscale("linear")
    plt.yscale("linear")
    plt.show()

fila = 5
columna = 6

features_fila_col = X_train[:, fila, columna]
print(len(np.unique(features_fila_col)))

title = "Valores en (" + str(fila) + ", " + str(columna) + ")"
plot_X(features_fila_col, title, fila, columna)
"""




##############################--Aqui empieza el código de la práctica--##############################

def load_MNIST_for_adaboost(clase, tipo):
    # Cargar los datos de entrenamiento y test tal y como nos los sirve keras (MNIST de Yann Lecun)
    (X_train, Y_train), (X_test, Y_test) = keras.datasets.mnist.load_data()

    # Formatear imágenes a vectores de floats y normalizar
    X_train = X_train.reshape((X_train.shape[0], 28*28)).astype("float32") / 255.0
    X_test = X_test.reshape((X_test.shape[0], 28*28)).astype("float32") / 255.0

    if tipo == "Binario":
        # Filtrar las imágenes de la clase deseada
        Y_train = (Y_train == clase).astype(int)*2 - 1
        Y_test = (Y_test == clase).astype(int)*2 - 1 
        return X_train, Y_train, X_test, Y_test
    
    if tipo == "Multiclase":
        Y_train_vec = []
        Y_test_vec = []
        for i in range(clase):
            Y_train_vec.append((Y_train == i).astype(int)*2 - 1)
            Y_test_vec.append((Y_test == i).astype(int)*2 - 1)
        return X_train, Y_train_vec, X_test, Y_test_vec, Y_train, Y_test


##DECISION STUMP
class DecisionStump:
    ## Constructor de clase, con número de características
    def __init__(self, n_features):
        # Seleccionar al azar una característica, un umbral y una polaridad.
        self.caracteristica = random.randint(0, n_features - 1)
        self.umbral = np.random.rand()
        self.polaridad = np.random.choice([-1, 1])


    ## Método para obtener una predicción con el clasificador débil
    def predict(self, X):
        # Si la característica que comprueba este clasificador es mayor que el umbral y la polaridad es 1
        # o si es menor que el umbral y la polaridad es -1, devolver 1 (pertenece a la clase)
        # Si no, devolver -1 (no pertenece a la clase)
        caracteristicas = X[:, self.caracteristica]
        predicciones = np.where(self.polaridad * caracteristicas > self.polaridad * self.umbral, 1, -1)
        return predicciones
                
        
        

##ADABOOST
class Adaboost:
    ## Constructor de clase, con número de clasificadores e intentos por clasificador
    def __init__(self, T=5, A=20):
        # Dar valores a los parámetros del clasificador e iniciar la lista de clasificadores débiles vacía
        self.T = T
        self.A = A
        self.classifiers = []
        self.alphas = []

    
    ## Método para entrenar un clasificador fuerte a partir de clasificadores débiles mediante Adaboost
    def fit(self, X, Y, verbose = False):
        # Obtener el número de observaciones y de características por observación de X
        n_observaciones, n_caracteristicas = X.shape
        # Iniciar pesos de las observaciones a 1/n_observaciones
        pesos = np.ones(n_observaciones) / n_observaciones
        # Bucle de entrenamiento Adaboost: desde 1 hasta T repetir
        for t in range(self.T):
            mejorError = 1
            mejorClasificador = ""
            mejorPrediccion = ""

            clasificadoresT = []
            erroresT = []
            alphasT = []

            # Bucle de búsqueda de un buen clasificador débil: desde 1 hasta A repetir
            for a in range(self.A):
                # Crear un nuevo clasificador débil aleatorio
                clasificadorDebil = DecisionStump(n_caracteristicas)
                clasificadoresT.append(clasificadorDebil)

                # Calcular predicciones de ese clasificador para todas las observaciones
                prediccion = clasificadorDebil.predict(X)

                # Calcular el error: comparar predicciones con los valores deseados y acumular los pesos de las observaciones mal clasificadas
                error = np.sum(pesos * (prediccion != Y))
                erroresT.append(error)

                # Actualizar mejor clasificador hasta el momento: el que tenga menor error
                if error < mejorError:
                    mejorClasificador = clasificadorDebil
                    mejorPrediccion = prediccion
                    mejorError = error
                
            # Calcular el valor de alfa y las predicciones del mejor clasificador débil
            alpha = 0.5 * np.log2((1 - mejorError) / max(mejorError, 1e-10))
            alphasT.append(alpha)

            # Actualizar pesos de las observaciones en función de las predicciones, los valores deseados y alfa
            pesosAnt = np.sum(pesos)
            pesos = pesos * np.exp(-alpha * Y * mejorClasificador.predict(X))

            # Normalizar a 1 los pesos
            pesos = pesos / pesosAnt

            # Guardar el clasificador y el alpha en la lista de clasificadores de Adaboost
            self.classifiers.append(mejorClasificador)
            self.alphas.append(alpha)

            # Imprimir información de depuración si verbose es True
            if verbose: 
                mensaje = "Añadido clasificador {:>3}: {:>4}, {:>6.4f}, {:+2}, {:>8.6f}".format(t+1, mejorClasificador.caracteristica, mejorClasificador.umbral, mejorClasificador.polaridad, mejorError)
                print(mensaje)
            
            #if verbose:
            #    mensaje = "Iteración {:>3}/{:>3}, Error: {:>8.6f}".format(t+1, self.T, mejorError)
    
    ## Método para obtener una predicción con el clasificador fuerte Adaboost
    def predict(self, X, tipoAdaboost):
        # Calcular las predicciones de cada clasificador débil para cada input multiplicadas por su alfa
        # Sumar para cada input todas las predicciones ponderadas y decidir la clase en función del signo
        if tipoAdaboost == "Binario":
            predicciones = np.array([alpha * clasificador.predict(X) for alpha, clasificador in zip(self.alphas, self.classifiers)])
            predicciones = np.sign(np.sum(predicciones, axis=0))

        if tipoAdaboost == "Multiclase":
            predicciones = np.sum([alpha * clasificador.predict(X) for alpha, clasificador in zip(self.alphas, self.classifiers)], axis=0)
            

        return predicciones

def precision(Y, Y_pred):
    # Calcular la precisión como el número de aciertos sobre el total de predicciones
    precision = np.sum(Y == Y_pred) / len(Y)
    return precision

## Métodos para las tareas de la práctica

def tareas_1A_y_1B_adaboost_binario(clase, T, A, verbose=False):
    if verbose:
        print("****Inicio de tareas_1A_y_1B_adaboost_binario****")
    # Cargar los datos de entrenamiento y test tal y como nos los sirve keras (MNIST de Yann Lecun) para la clase deseada
    X_train, Y_train, X_test, Y_test = load_MNIST_for_adaboost(clase, "Binario")


    #Crea un clasificador Adaboost con los parámetros T y A y lo entrena con los datos de entrenamiento
    adaboost = Adaboost(T, A)
    if verbose:
        print("Entrenando clasificador Adaboost para el digito = " + str(clase) + " con T = " + str(T) + " y A = " + str(A) + "...")
    start = time.time()
    adaboost.fit(X_train, Y_train)
    end = time.time()
    total_time = end - start

    if verbose:
        print("Tiempo de entrenamiento: {:.2f} segundos".format(total_time))
    
    # Obtener las predicciones del clasificador Adaboost para los datos de entrenamiento y test
    y_train_pred = adaboost.predict(X_train, "Binario")
    y_test_pred = adaboost.predict(X_test, "Binario")

    # Calcular la precisión para los datos de entrenamiento y test
    y_train_accuracy = precision(Y_train, y_train_pred)
    y_test_accuracy = precision(Y_test, y_test_pred)

    # Imprimir las tasas de acierto
    if verbose:
        print("Tasa de acierto (train, test) y tiempo: {:.2f}%, {:.2f}%, {:.3f} s.".format(y_train_accuracy * 100, y_test_accuracy * 100, total_time))
    
    if verbose:
        print("****Fin de tareas_1A_y_1B_adaboost_binario****\n")

    return y_test_accuracy, total_time

def tarea_1C_graficas_rendimiento(rend_1A):
    print("****Inicio de tarea_1C_graficas_rendimiento****")
    valores_T = [1, 2, 5, 10, 15, 20, 25, 30, 45]
    valores_A = [1, 2, 5, 10, 15, 20, 25, 30, 45]
    
    tasa_acierto_train = []
    tiempos_train = []

    for T in valores_T:
        tasa_acierto_aux = []
        tiempos_aux = []

        for A in valores_A:
            rendimiento, tiempo = tareas_1A_y_1B_adaboost_binario(clase=9, T=T, A=A, verbose=False)
            tasa_acierto_aux.append(rendimiento)
            tiempos_aux.append(tiempo)
        
        tiempos_train.append(np.mean(tiempos_aux))
        tasa_acierto_train.append(np.mean(tasa_acierto_aux))

    fig, ax1 = plt.subplots()

    # Configurar el eje izquierdo (tasa de acierto)
    color = 'tab:blue'
    ax1.set_xlabel('Valores de T')
    ax1.set_ylabel('Tasa de Acierto', color=color)
    ax1.plot(valores_T, tasa_acierto_train, color=color, label='Tasa de Acierto')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc='upper left')

    # Crear el eje derecho (tiempo)
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Tiempo', color=color)
    ax2.plot(valores_T, tiempos_train, color=color, label='Tiempo')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='upper right')

    # Ajustar el diseño y mostrar el gráfico
    fig.tight_layout()
    plt.title('Gráfica de Tasa de Acierto y Tiempo')
    plt.savefig('Grafica_Entrenamiento.png')
    print("\nSe ha guardado la gráfica de entrenamiento en el archivo Grafica_Entrenamiento.png")
    print("Se han establecido los valores de T=425 y A=2 como los óptimos\n")
    t_optimo = 450
    a_optimo = 2

    print("****Fin de tarea_1C_graficas_rendimiento****\n")

    return t_optimo, a_optimo

def tarea_1D_adaboost_multiclase(T, A, verbose=False):
    if verbose:
        print("****Inicio de tarea_1D_adaboost_multiclase****")
    #Se definen todas las clases
    clases = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    entrenoAdaboost = []

    # Cargar los datos de entrenamiento y test tal y como nos los sirve keras (MNIST de Yann Lecun) para todas las clases
    X_train, Y_train_vec, X_test, Y_test_vec, Y_train, Y_test = load_MNIST_for_adaboost(10, "Multiclase")

    start = time.time()
    for clase in clases:
        #Entrenar adaboost para ese dígito
        adaboost = Adaboost(T, A)
        if verbose:
            print("Entrenando clasificador Adaboost para el digito = " + str(clase) + " con T = " + str(T) + " y A = " + str(A))
        
        adaboost.fit(X_train, Y_train_vec[clase])
        entrenoAdaboost.append(adaboost)
    
    end = time.time()
    total_time = end - start

    # Obtener las predicciones del clasificador Adaboost para los datos de entrenamiento y test
    y_train_pred = np.argmax(np.array([adaboost.predict(X_train, "Multiclase") for adaboost in entrenoAdaboost]), axis=0)
    y_test_pred = np.argmax(np.array([adaboost.predict(X_test, "Multiclase") for adaboost in entrenoAdaboost]), axis=0)

    # Calcular las tasas de acierto para los datos de entrenamiento y test
    y_train_accuracy = precision(Y_train, y_train_pred)
    y_test_accuracy = precision(Y_test, y_test_pred)

    # Imprimir las tasas de acierto
    if verbose:
        print("Tasa de acierto (train, test) y tiempo: {:.2f}%, {:.2f}%, {:.3f} s.".format(y_train_accuracy * 100, y_test_accuracy * 100, total_time))
    
    if verbose:
        print("****Fin de tarea_1D_adaboost_multiclase****\n")
    
    return y_test_accuracy, total_time
    
def tarea_2A_AdaBoostClassifier_default(n_estimators, verbose=False):
    if verbose:
        print("****Inicio de tarea_2A_AdaBoostClassifier_default****")

    # Cargar el conjunto de datos MNIST
    X_train, Y_train_vec, X_test, Y_test_vec, Y_train, Y_test = load_MNIST_for_adaboost(10, "Multiclase")

    # Tomar un subconjunto más pequeño del conjunto de datos para acelerar el tiempo de ejecución
    n_samples = 2000  
    X_train_subset = X_train[:n_samples]
    y_train_subset = Y_train[:n_samples]
    
    # Escalar los datos (es importante para SVM)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_subset)
    X_test_scaled = scaler.transform(X_test)
    
    # Crear un clasificador base (SVM lineal en este caso)
    base_classifier = SVC(kernel='linear', probability=True, C=1.0)
    
    # Crear el clasificador Adaboost
    adaboost_classifier = AdaBoostClassifier(base_classifier, n_estimators=n_estimators, random_state=42)

    # Medir el tiempo de entrenamiento en el subconjunto
    start_time = time.time()
    adaboost_classifier.fit(X_train_scaled, y_train_subset)
    total_time = time.time() - start_time

    if verbose:
        print("Tiempo de entrenamiento: {:.2f} segundos".format(total_time))

    # Predecir las etiquetas del conjunto de prueba
    y_train_test = adaboost_classifier.predict(X_train_scaled)
    y_test_pred = adaboost_classifier.predict(X_test_scaled)

    # Calcular la precisión del clasificador
    y_train_accuracy = accuracy_score(y_train_subset, y_train_test)
    y_test_accuracy = accuracy_score(Y_test, y_test_pred)

    if verbose:
        print("Tasa de acierto (train, test) y tiempo: {:.2f}%, {:.2f}%, {:.3f} s.".format(y_train_accuracy * 100, y_test_accuracy * 100, total_time))

        print("****Fin de tarea_2A_AdaBoostClassifier_default****\n")

    return y_test_accuracy, total_time

def tarea_2B_graficas_rendimiento(rend_1D, rend_2A):
    print("****Inicio de tarea_2B_graficas_rendimiento****")
    valores_T = [10, 20, 40]
    valores_A = [10, 20, 40]

    for i in range (2):
        tasa_acierto = []
        tiempos = []

        if i == 0:
            for T in valores_T:
                tasa_acierto_aux = []
                tiempos_aux = []

                for A in valores_A:
                    rendimiento, tiempo = tarea_1D_adaboost_multiclase(T=T, A=A, verbose=False)
                        
                    tasa_acierto_aux.append(rendimiento)
                    tiempos_aux.append(tiempo)
                
                tiempos.append(np.mean(tiempos_aux))
                tasa_acierto.append(np.mean(tasa_acierto_aux))
        
        if i == 1:
            for A in valores_A:
                rendimiento, tiempo = tarea_2A_AdaBoostClassifier_default(n_estimators=A)

                tasa_acierto.append(rendimiento)
                tiempos.append(tiempo)

        fig, ax1 = plt.subplots()

        # Configurar el eje izquierdo (tasa de acierto)
        color = 'tab:blue'
        ax1.set_xlabel('Valores de T')
        ax1.set_ylabel('Tasa de Acierto', color=color)
        ax1.plot(valores_T, tasa_acierto, color=color, label='Tasa de Acierto')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.legend(loc='upper left')

        # Crear el eje derecho (tiempo)
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Tiempo', color=color)
        ax2.plot(valores_T, tiempos, color=color, label='Tiempo')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.legend(loc='upper right')

        # Ajustar el diseño y mostrar el gráfico
        fig.tight_layout()
        plt.title('Gráfica de Tasa de Acierto y Tiempo')

        if i == 0:
            plt.savefig('Grafica_Test_1D.png')
            print("Se ha guardado la gráfica de test en el archivo Grafica_Test_1D.png")
        
        if i == 1:
            plt.savefig('Grafica_Test_2A.png')
            print("Se ha guardado la gráfica de test en el archivo Grafica_Test_2A.png")


    print("****Fin de tarea_2B_graficas_rendimiento****\n")

    return True

def tarea_2C_AdaBoostClassifier_faster(n_estimators, verbose=False):
    if verbose:
        print("****Inicio de tarea_2C_AdaBoostClassifier_faster****")
    
    X_train, Y_train_vec, X_test, Y_test_vec, Y_train, Y_test = load_MNIST_for_adaboost(10, "Multiclase")
    X_train, X_val, Y_train, y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)

    # Configurar el clasificador base (Decision Tree)
    base_classifier = DecisionTreeClassifier(max_depth=2)

    # Configurar el clasificador Adaboost
    adaboost_classifier = AdaBoostClassifier(base_classifier, n_estimators=n_estimators, random_state=42)

    # Entrenar el modelo
    start_time = time.time()
    adaboost_classifier.fit(X_train, Y_train)
    total_time = time.time() - start_time

    if verbose:
        print("Tiempo de entrenamiento: {:.2f} segundos".format(total_time))

    # Realizar predicciones
    y_train_pred = adaboost_classifier.predict(X_val)
    y_test_pred = adaboost_classifier.predict(X_test)

    # Calcular la precisión
    accuracy_train = accuracy_score(y_val, y_train_pred)
    accuracy_test = accuracy_score(Y_test, y_test_pred)

    if verbose:
        print("Tasa de acierto (train, test) y tiempo: {:.2f}%, {:.2f}%, {:.3f} s.".format(accuracy_train * 100, accuracy_test * 100, total_time))

    if verbose:
        print("****Fin de tarea_2C_AdaBoostClassifier_faster****\n")

    return accuracy_test, total_time

def tarea_2C_graficas_rendimiento(rend_2A, rend_2C, calculado_2A=True):
    print("****Inicio de tarea_2C_graficas_rendimiento****")
    valores_T = [10, 20, 40]
    valores_A = [10, 20, 40]

    if calculado_2A == True:
        print("Se ha calculado la gráfica de test de 2A anteriormente, no se vuelve a calcular")
    else:
        tasa_acierto = []
        tiempos = []
        for A in valores_A:
            rendimiento, tiempo = tarea_2A_AdaBoostClassifier_default(n_estimators=A)

            tasa_acierto.append(rendimiento)
            tiempos.append(tiempo)
        
        fig, ax1 = plt.subplots()

        # Configurar el eje izquierdo (tasa de acierto)
        color = 'tab:blue'
        ax1.set_xlabel('Valores de T')
        ax1.set_ylabel('Tasa de Acierto', color=color)
        ax1.plot(valores_T, tasa_acierto, color=color, label='Tasa de Acierto')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.legend(loc='upper left')

        # Crear el eje derecho (tiempo)
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Tiempo', color=color)
        ax2.plot(valores_T, tiempos, color=color, label='Tiempo')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.legend(loc='upper right')

        # Ajustar el diseño y mostrar el gráfico
        fig.tight_layout()
        plt.title('Gráfica de Tasa de Acierto y Tiempo')

        plt.savefig('Grafica_Test_2A.png')
        print("Se ha guardado la gráfica de test en el archivo Grafica_Test_2A.png")
    
    tasa_acierto = []
    tiempos = []
    for A in valores_A:
        rendimiento, tiempo = tarea_2C_AdaBoostClassifier_faster(n_estimators=A)

        tasa_acierto.append(rendimiento)
        tiempos.append(tiempo)
    
    fig, ax1 = plt.subplots()

    # Configurar el eje izquierdo (tasa de acierto)
    color = 'tab:blue'
    ax1.set_xlabel('Valores de T')
    ax1.set_ylabel('Tasa de Acierto', color=color)
    ax1.plot(valores_T, tasa_acierto, color=color, label='Tasa de Acierto')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc='upper left')

    # Crear el eje derecho (tiempo)
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Tiempo', color=color)
    ax2.plot(valores_T, tiempos, color=color, label='Tiempo')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='upper right')

    # Ajustar el diseño y mostrar el gráfico
    fig.tight_layout()
    plt.title('Gráfica de Tasa de Acierto y Tiempo')

    plt.savefig('Grafica_Test_2C.png')
    print("Se ha guardado la gráfica de test en el archivo Grafica_Test_2C.png")
    
    print("***Fin de tarea_2C_graficas_rendimiento***\n")

def tarea_2D_MLP_Keras():
    print("****Inicio de tarea_2D_MLP_Keras****")
    X_train, Y_train_vec, X_test, Y_test_vec, Y_train, Y_test = load_MNIST_for_adaboost(10, "Multiclase")
    Y_train = to_categorical(Y_train, 10)
    Y_test = to_categorical(Y_test, 10)

    # Construir el modelo MLP
    model = Sequential()
    model.add(Dense(256, activation='relu', input_shape=(784,)))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    # Compilar el modelo
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Entrenar el modelo
    start_time = time.time()
    model.fit(X_train.reshape(-1, 784), Y_train, epochs=10, batch_size=128, validation_split=0.2)
    total_time = time.time() - start_time

    # Evaluar el modelo en el conjunto de prueba
    loss, accuracy = model.evaluate(X_test.reshape(-1, 784), Y_test)
    print("Tasa de acierto y tiempo: {:.2f}%, {:.3f} s.".format(accuracy * 100, total_time))

    print("***Fin de tarea_2D_MLP_Keras***\n")

    return accuracy, total_time


#Main
if __name__ == "__main__":
    ## Las llamadas a funciones auxiliares que sean relevantes para algo
    ## en la evaluación pueden dejarse comentadas en esta sección.
    # test_DecisionStump(9, 59, 0.4354, 1)

    ## Tarea 1A y 1B
    rend_1A = tareas_1A_y_1B_adaboost_binario(clase=9, T=15, A=15, verbose=True)
    
    ## Tarea 1C
    ## Una parte de la tarea 1C es fijar los parámetro más adecuados    
    valorT, valorA = tarea_1C_graficas_rendimiento(rend_1A)
    
    ## Tarea 1D
    rend_1D = tarea_1D_adaboost_multiclase(T=valorT, A=valorA, verbose=True)

    ## Tarea 2A
    estimadores = 2
    rend_2A = tarea_2A_AdaBoostClassifier_default(n_estimators=estimadores, verbose=True)

    ## Tarea 2B
    ## Esto se encuentra comentado ya que requiere mucho tiempo realizar las 
    ## gráficas. Si se quieren obtener, descomentar la siguiente línea
    ## Las gráficas obtenidas por mi se encuentran en la documentación/informe/memoria
    calculado_2A = False
    calculado_2A = tarea_2B_graficas_rendimiento(rend_1D, rend_2A)

    ## Tarea 2C
    estimadores=50
    rend_2C = tarea_2C_AdaBoostClassifier_faster(n_estimators=estimadores, verbose=True)

    ## Esto se encuentra comentado ya que requiere mucho tiempo realizar las 
    ## gráficas. Si se quieren obtener, descomentar la siguiente línea
    ## Las gráficas obtenidas por mi se encuentran en la documentación/informe/memoria
    tarea_2C_graficas_rendimiento(rend_2A, rend_2C, calculado_2A=calculado_2A)

    ## Tarea 2D
    rend_2D = tarea_2D_MLP_Keras()