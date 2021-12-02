import gzip
import numpy as np
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
import numpy as np


def open_images(filename):
    with gzip.open(filename, "rb") as file:
        data = file.read()
        return np.frombuffer(data, dtype=np.uint8, offset=16) \
            .reshape(-1, 28, 28) \
            .astype(np.float32)


def open_labels(filename):
    with gzip.open(filename, "rb") as file:
        data = file.read()
        return np.frombuffer(data, dtype=np.uint8, offset=8)

X_train = open_images("data/mnist/train-images-idx3-ubyte.gz")
y_train = open_labels("data/mnist/train-labels-idx1-ubyte.gz")


# print("X_train[2]")
# print(type(X_train[2]))
# print(len(X_train[2]))
# print(X_train[2])
# plt.imshow(X_train[2], cmap='gray_r')
# plt.show()

# print(X_train[11])
# plt.imshow(X_train[11], cmap='gray_r')
# plt.show()

# print("y_train[0]", y_train[0])
# print("y_train[1]", y_train[1])
# print("y_train[11]", y_train[11])
# print("y_train", y_train)


def bestimmeReferenzNummer(i):
    global y_train
    y_train = y_train == i  # Vergleich ueber alle Elemente im Daten-Array, Aufloesung nach T-Shirt == 0


bestimmeReferenzNummer(5)

model = Sequential()
# hiddenlayer
model.add(Dense(100, activation="sigmoid", input_shape=(784,)))
# outputlayer
model.add(Dense(1, activation="sigmoid"))
model.compile(optimizer="sgd", loss="binary_crossentropy")
a = X_train.reshape(60000, 784)
model.fit(a, y_train, epochs=50, batch_size=1000)


def predictAndShow(i):
    # print("\nImage ", i, " is a 5?\t ", y_train[i])
    plt.imshow(i, cmap='gray_r')
    plt.show()
    print("probability:\t\t\t ", model.predict(i.reshape(1, 784)))
    if ((model.predict(i.reshape(1, 784))) > (0.6)):
        print("TRUE")
    else:
        print("FALSE")


def imageprepare(argv):
    """
    This function returns the pixel values.
    The imput is a png file location.
    """
    im = Image.open(argv).convert('L')
    width = float(im.size[0])
    height = float(im.size[1])
    newImage = Image.new('L', (28, 28), (255))  # creates white canvas of 28x28 pixels

    if width > height:  # check which dimension is bigger
        # Width is bigger. Width becomes 20 pixels.
        nheight = int(round((20.0 / width * height), 0))  # resize height according to ratio width
        if (nheight == 0):  # rare case but minimum is 1 pixel
            nheight = 1
            # resize and sharpen
        img = im.resize((20, nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((28 - nheight) / 2), 0))  # calculate horizontal position
        newImage.paste(img, (4, wtop))  # paste resized image on white canvas
    else:
        # Height is bigger. Heigth becomes 20 pixels.
        nwidth = int(round((20.0 / height * width), 0))  # resize width according to ratio height
        if (nwidth == 0):  # rare case but minimum is 1 pixel
            nwidth = 1
            # resize and sharpen
        img = im.resize((nwidth, 20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((28 - nwidth) / 2), 0))  # caculate vertical pozition
        newImage.paste(img, (wleft, 4))  # paste resized image on white canvas

    # newImage.save("sample.png

    tv = list(newImage.getdata())  # get pixel values

    # normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
    tva = [(255 - x) * 1.0 / 255.0 for x in tv]
    print(tva)
    return tva


x = imageprepare('../image/five.png')  # file path here
y = np.array_split(x, 28)
z = np.array(y)
print("z")
print("z type", type(z))
print(len(z))  # mnist IMAGES are 28x28=784 pixels
print(z)

# import matplotlib.pyplot as plt
# plt.imshow(y, cmap='gray_r')
# plt.show()


# predictAndShow(X_train[0])
# predictAndShow(X_train[1])
# predictAndShow(X_train[11])
predictAndShow(X_train[0])
predictAndShow(z)
