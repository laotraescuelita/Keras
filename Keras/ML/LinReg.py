
#Vamos a probar la data de boston_housing 
#Keras necesita estar conectado a internet para leer la data
from keras.datasets import boston_housing
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

#Mostramos el tipo y el tamaño de las matrices a entrenar
print("Type of the Dataset:",type(y_train))
print("Shape of training data :",x_train.shape)
print("Shape of training labels :",y_train.shape)
print("Shape of testing data :",type(x_test))
print("Shape of testing labels :",y_test.shape)

#Aquí verificamos que los datos son de tipo numerico.
print( x_train[:3,:] )
#Normalmente se necista limpiar, transformar y reducir los datos, 
#pero estos son datos ya listos para probar los modelos. 

#Comencemos a crear la arquitectura
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
#En keras es necesario tener 3 conjuntos de datos, train, test, validate
x_val = x_train[300:,]
y_val = y_train[300:,]

#Las capas van pasando de 13 a 6 y al final 1.
#Hay dos funciones de activacion, pero no hay en la salida. 
#Aquí las dimensiones hacen refernecia a las columnas que se ingresan.
model = Sequential()
model.add(Dense(13, input_dim=13, kernel_initializer='normal', activation='relu'))
model.add(Dense(6, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal'))

#La función de costo es cambia con respesto a los problemas de clasificación.
#La función del gradiente es adam, pero podemos seleccionar stochastic gradient descent.
#LA medición del error también cambio con respesto al ejemplo de clasificación.
model.compile(loss='mean_squared_error', optimizer='adam', 
metrics=['mean_absolute_percentage_error']) 

#Entrena el modelo
model.fit(x_train, y_train, batch_size=32, epochs=3, 
validation_data=(x_val,y_val))  #Aquí podemos modificar las iteraciones.

#Evaluemos el modelo con la matriz y vector de test.
results = model.evaluate(x_test, y_test)
for i in range(len(model.metrics_names)):
	print(model.metrics_names[i]," : ", results[i])
