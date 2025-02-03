#importing libraries
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import utils

#loading data and splitting them by train and test
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

#classifying clothes and normalizing them (0,1)
class_names = ['T-shirt/top', 'Trousers', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
x_train = x_train / 255
x_test = x_test / 255

#creating a model with 3 layers flattening them by vector of 28:28
#128 neuron layer with ReLU activation function
#10 (10 classes) with Softmax activation function
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

#compiling model with Adam optimizer with sparse_categorical_crossentropy loss focusing on accuracy
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#showing model
model.summary()

#fitting model data with 10 epochs and evaluating them
model.fit(x_train, y_train, epochs=10)

#2 variables of loss and accuracy
test_loss, test_acc = model.evaluate(x_test, y_test)

print('Test accuracy:', test_acc)

#making predictions already
predictions = model.predict(x_train)

print("Prediction: ", np.argmax(predictions[5]), "Right Answer: ", class_names[y_train[5]])
