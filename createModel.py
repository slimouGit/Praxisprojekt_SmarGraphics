import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

class PredictiveModel():
    def __init__(self, columns:int, epoche:int, reference:int):
        print("Model")
        bestimmeReferenzNummer(reference)
        initializeModel(columns,epoche)

x_test = np.load('digits_x_test.npy')
y_test = np.load('digits_y_test.npy')
x_test.shape, y_test.shape


def bestimmeReferenzNummer(i):
    global y_test
    y_test = y_test == i

#--------------------------------------------------------------

def initializeModel(columns:int, epoche:int):
    global model
    model = Sequential()
    # hiddenlayer
    model.add(Dense(100, activation="sigmoid", input_shape=(784,)))
    # outputlayer
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer="sgd", loss="binary_crossentropy")
    a = x_test.reshape(columns * 10, 784)
    model.fit(a, y_test, epochs=epoche, batch_size=1000)


PredictiveModel(21, 5000, 1)

