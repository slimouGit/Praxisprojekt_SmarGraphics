import config
import image_converter
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from keras.models import Sequential
from keras.layers import Dense

#--------------------------------------------------------------

image = Image.open(config.filename)
predictedValues = []
possibleMatches = []

#--------------------------------------------------------------

def show(img, figsize=(8, 4), title=None):
    plt.figure(figsize=figsize)
    plt.imshow(img)
    if title:
        plt.title(title)
    plt.show()

show(image)

#--------------------------------------------------------------

x_test = np.load(config.xfile)
y_test = np.load(config.yfile)
x_test.shape, y_test.shape

#--------------------------------------------------------------

def bestimmeReferenzNummer(i):
    global y_test
    y_test = y_test == i  # Vergleich ueber alle Elemente im Daten-Array, Aufloesung nach T-Shirt == 0

bestimmeReferenzNummer(config.reference)

#--------------------------------------------------------------

def initializeModel():
    global model
    model = Sequential()
    model.add(Dense(config.hidden_layer, activation="sigmoid", input_shape=(784,)))
    # outputlayer
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer="sgd", loss="binary_crossentropy")
    a = x_test.reshape(config.columns * 10, 784)
    model.fit(a, y_test, epochs=config.epoche, batch_size=config.batch)

initializeModel()

#--------------------------------------------------------------

def predictAndShow(expected, img):
    plt.imshow(img)
    plt.show()
    print("probability of predicting", config.reference ," by image with number ", expected, " is \t\t\t ", model.predict(img.reshape(1, 784)))
    if( model.predict(img.reshape(1, 784))>config.border):
        print("POSSIBLE")
    predictedValues.append(model.predict(img.reshape(1, 784)))

#--------------------------------------------------------------

def determinePredictedNumber(predictedValues):
    max_value = max(predictedValues)
    max_index = predictedValues.index(max_value)
    if(max_value<(config.border*0.1)):
        print("It seems, searched number ", config.reference, "is not found")
    else:
        print("probably searched number ", config.reference, " was found at index ", max_index)

#--------------------------------------------------------------

def determinePossibleMatches(possibleMatches):
    if (len(possibleMatches) == 0):
        print("looks like, no number is ", config.reference)
    if (len(possibleMatches) == 1):
        print("index with high probability to be number is", possibleMatches)
    if(len(possibleMatches) > 1):
        max_value = max(possibleMatches)
        max_index = possibleMatches.index(max_value)
        print("index with high probability to be number ", config.reference, " is ", max_index)

#--------------------------------------------------------------

print("-------------------- IMAGES --------------------")
print("REFERENCE NUMBER ", config.reference)
predictAndShow(0, image_converter.read_image('image/null.png'))
predictAndShow(1, image_converter.read_image('image/one.png'))
predictAndShow(2, image_converter.read_image('image/two.png'))
# predictAndShow(3, image_converter.read_image('image/three.png'))
# predictAndShow(4, image_converter.read_image('image/four.png'))
predictAndShow(5, image_converter.read_image('image/five.png'))
# predictAndShow(6, image_converter.read_image('image/six.png'))
predictAndShow(7, image_converter.read_image('image/seven.png'))
predictAndShow(8, image_converter.read_image('image/eight.png'))
predictAndShow(9, image_converter.read_image('image/nine.png'))

#--------------------------------------------------------------

determinePredictedNumber(predictedValues)

#--------------------------------------------------------------

