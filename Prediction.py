import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter
from keras.models import Sequential
from keras.layers import Dense

x_test = np.load('digits_x_test.npy')
y_test = np.load('digits_y_test.npy')
x_test.shape, y_test.shape
global model
model = Sequential()
mylist = []





class PredictiveModel():

    def __init__(self, columns:int, epoche:int, reference:int):
        print("Model ", reference)

        global y_test
        y_test = y_test == reference

        initializeModel(y_test, columns, epoche)

        predictAndShow(readImage('image/five.png'))


def initializeModel(y, columns:int, epoche:int):

    model.add(Dense(100, activation="sigmoid", input_shape=(784,)))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer="sgd", loss="binary_crossentropy")
    a = x_test.reshape(columns * 10, 784)
    model.fit(a, y, epochs=epoche, batch_size=1000)

def imageprepare(argv):
    im = Image.open(argv).convert('L')
    width = float(im.size[0])
    height = float(im.size[1])
    newImage = Image.new('L', (28, 28), (255))
    if width > height:
        nheight = int(round((28.0 / width * height), 0))
        if (nheight == 0):
            nheight = 1
        img = im.resize((28, nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((28 - nheight) / 2), 0))
        newImage.paste(img, (0, wtop))
    else:
        nwidth = int(round((28.0 / height * width), 0))
        if (nwidth == 0):
            nwidth = 1
        img = im.resize((nwidth, 28), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((28 - nwidth) / 2), 0))
        newImage.paste(img, (wleft, 0))
    tv = list(newImage.getdata())
    tva = [(255 - x) * 1.0 / 255.0 for x in tv]
    return tva

def readImage(img):
    x_1 = imageprepare(img)
    y_1 = np.array_split(x_1, 28)
    return np.array(y_1)

def predictAndShow(img):
    plt.imshow(img)
    plt.show()
    mylist.append(model.predict(img.reshape(1, 784)))

class IterateModel():
    def __init__(self):
        for i in range(10):
            PredictiveModel(21, 1, i)

IterateModel()

def calculatePrediction(mylist):
    max_value = max(mylist)
    max_index = mylist.index(max_value)
    print("myList ", mylist)
    print("predicted number is ", max_index)

calculatePrediction(mylist)

