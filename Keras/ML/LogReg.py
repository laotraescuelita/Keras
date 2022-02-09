#Vamos a genrar datos sinteticos con las librerias de sklearn. 
from sklearn.datasets import make_classification 
features, target = make_classification(n_samples = 100,
 n_features = 10,
 n_informative = 5,
 n_redundant = 0,
 n_classes = 2,
 weights = [.25, .75],
 random_state = 1)

#En keras hay que crear tres conjuntos de matrices y vectores. Train, validate, test.
x_train = features[:60]
y_train = target[:60]

x_val = features[40:60]
y_val = target[40:60]

x_test = features[80:]
y_test = target[80:]

 
#Vamos a preparar la arquitectura del modelo.
from keras.models import Sequential
from keras.layers import Dense, Activation 

#Aquí se escojen las funciones de activación. Relu funciona en la capas escondias, 
#pero debe ser sigmoide en la ultima capa para que clasifique.
model = Sequential()
model.add(Dense(64, input_dim=10,activation = "relu")) #Layer 1
model.add(Dense(32,activation = "relu")) #Layer 2
model.add(Dense(16,activation = "relu")) #Layer 3
model.add(Dense(8,activation = "relu")) #Layer 4
model.add(Dense(4,activation = "relu")) #Layer 5
model.add(Dense(1,activation = "sigmoid")) #Output Layer

#Aquí se selecciona la funcion de costo y el gradiente. 
model.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['accuracy'])

#Entrenemos el modelo
model.fit(x_train, y_train, batch_size=64, epochs=3, validation_data=(x_val,y_val))
print( "[============Evaluating the model ==================]")
print(model.evaluate(x_test,y_test))
print(model.metrics_names)
pred = model.predict(x_test)
#Vamos a imprimir los resltados y los comparamos con los reales. 
#Aqui arriba de 0.5 sera 1 y menor 0.
print(pred[:5] , y_test[:5]  )
