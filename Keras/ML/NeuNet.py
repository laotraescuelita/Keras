
#Importar las librerias que se utilizaran
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
#Esta código lo obtuve del canal python-enginer, lo que hace es desactivar algunso mensajes de error.
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


#Paso cero: Preprocesamiento no aplica porque la matrices ya estan listas para ser entrenadas.

#Importar la base de mnist, para reconocer digitos.
libreria = keras.datasets.mnist
#Dividri la matriz, una parte para entrenamiento otra para pruebas.
(X_train, y_train), (X_test, y_test) = libreria.load_data()
print(X_train.shape, y_train.shape)

#Paso uno: Técnicas de reducción de matrices, estandarizar, normlaizar, codificar, etc...
#Normalizar los datos para que estene entre 1 y 0.
X_train, X_test = X_train / 255.0, X_test / 255.0


#Paso dos: Seleccionar el módelo que aplica a nuestro problema.
modelo = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28,28)), #Aquí debemos reducir la matriz
    keras.layers.Dense(128, activation='relu'), #Solo hay una capa oculta.
    keras.layers.Dense(10), # y una capa de salida con diez salidas.
])

print(modelo.summary())

# Para los que vamos empezando aquí esta el procedimiento paso a paso.
#modelo = keras.models.Sequential()
#modelo.add(keras.layers.Flatten(input_shape=(28,28))
#modelo.add(keras.layers.Dense(128, activation='relu'))
#modelo.add(keras.layers.Dense(10))

# Elegir función de costo y el algoritmo para el descenso del gradiente.
costo = keras.losses.SparseCategoricalCrossentropy(from_logits=True) 
gradiente = keras.optimizers.Adam(learning_rate=0.001) 
kpi = ["accuracy"]

modelo.compile(loss=costo, optimizer=gradiente, metrics=kpi)

# Tamaño del entrenamiento
lote = 64
iteraciones = 5

modelo.fit(X_train, y_train, batch_size=lote, epochs=iteraciones, shuffle=True, verbose=2)

# Evaluar.
modelo.evaluate(X_test, y_test, batch_size=lote, verbose=2)

# Predecir
# a) contruir el módelo con la función de costo sofmax
modelodos = keras.models.Sequential([modelo,keras.layers.Softmax()])
predicciones = modelodos(X_test)
prediccionuno = predicciones[0]
print(prediccionuno)
# La función argmax nos ayude a categorizar la predicción.
categoria = np.argmax(prediccionuno)
print(categoria)

# b) módelo + nn.softmax.
predicciones = modelo(X_test)
predicciones = tf.nn.softmax(predicciones)
prediccionuno = predicciones[0]
print(prediccionuno)
categoria = np.argmax(prediccionuno)
print(categoria)

# c) módelo + nn.softmax
predicciones = modelo.predict(X_test, batch_size=lote)
predicciones = tf.nn.softmax(predicciones)
prediccionuno = predicciones[0]
print(prediccionuno)
categoria = np.argmax(prediccionuno)
print(categoria)

# Hay que aplicar la función argmax para separar las variables categoricas.
prediccionvarias = predicciones[0:5]
print(prediccionvarias.shape)
categorias = np.argmax(prediccionvarias, axis=1)
print(categorias)

