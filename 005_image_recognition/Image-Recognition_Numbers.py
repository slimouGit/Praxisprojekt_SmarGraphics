import gzip
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

print("image: T-Shirt")
print(X_train[1])


#print(y_train)
y_train = y_train == 0 #Vergleich ueber alle Elemente im Daten-Array, Aufloesung nach T-Shirt == 0
#print(y_train)
# print(X_train)
# print(X_train.shape)

import matplotlib.pyplot as plt

# print(y_train)

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
#hiddenlayer
model.add(Dense(100, activation="sigmoid", input_shape=(784,)))
#outputlayer
model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer="sgd", loss="binary_crossentropy")

a = X_train.reshape(60000,784)

model.fit(a, y_train, epochs=10, batch_size=1000)

#-------------------------------------------------------
from PIL import Image, ImageFilter

def imageprepare(argv):
    im = Image.open(argv).convert('L')
    width = float(im.size[0])
    height = float(im.size[1])
    newImage = Image.new('L', (28, 28), (255))  # creates white canvas of 28x28 pixels

    if width > height:
        nheight = int(round((20.0 / width * height), 0))  # resize height according to ratio width
        if (nheight == 0):  # rare case but minimum is 1 pixel
            nheight = 1
        img = im.resize((20, nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((28 - nheight) / 2), 0))  # calculate horizontal position
        newImage.paste(img, (4, wtop))  # paste resized image on white canvas
    else:
        nwidth = int(round((20.0 / height * width), 0))  # resize width according to ratio height
        if (nwidth == 0):  # rare case but minimum is 1 pixel
            nwidth = 1
        img = im.resize((nwidth, 20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((28 - nwidth) / 2), 0))  # caculate vertical pozition
        newImage.paste(img, (wleft, 4))  # paste resized image on white canvas
    tv = list(newImage.getdata())  # get pixel values
    tva = [(255 - x) * 1.0 / 255.0 for x in tv]
    print(tva)
    return tva
#-------------------------------------------------------
x=imageprepare('../image/five.png')#file path here
print(len(x))# mnist IMAGES are 28x28=784 pixels
y = np.array_split(x, 28)
print(y)
# plt.imshow(y, cmap='gray_r')
# plt.show()
Xn_train = y
yn_train = y
#-------------------------------------------------------

# print("\nReference Number 0 is 5")
# plt.imshow(X_train[0], cmap='gray_r')
# plt.show()
#
# print("\nReference Number 1 is 0")
# plt.imshow(X_train[1], cmap='gray_r')
# plt.show()
#
# print("\nReference Number 2 is 4")
# plt.imshow(X_train[2], cmap='gray_r')
# plt.show()
#
# print("\nReference Number 3 is 1")
# plt.imshow(X_train[3], cmap='gray_r')
# plt.show()
#
# print("\nReference Number 4 is 9")
# plt.imshow(X_train[4], cmap='gray_r')
# plt.show()
#
# print("\nReference Number 5 is 2")
# plt.imshow(X_train[5], cmap='gray_r')
# plt.show()
#
# print("\nReference Number 7 is 3")
# plt.imshow(X_train[7], cmap='gray_r')
# plt.show()
#
# print("\nReference Number 11 is 6")
# plt.imshow(X_train[13], cmap='gray_r')
# plt.show()
#
# print("\nReference Number 12 is 7")
# plt.imshow(X_train[15], cmap='gray_r')
# plt.show()
#
# print("\nReference Number 17 is 8")
# plt.imshow(X_train[17], cmap='gray_r')
# plt.show()

print("\nTest prints 1")
print(X_train[0])
plt.imshow(X_train[0], cmap='gray_r')
plt.show()
print("\nTest prints 2")
print(Xn_train[0])
print("\nTest prints 3")
u = np.array([Xn_train[0], Xn_train[1], Xn_train[2], Xn_train[3], Xn_train[4], Xn_train[5], Xn_train[6],
              Xn_train[7], Xn_train[8], Xn_train[9], Xn_train[10], Xn_train[11], Xn_train[12], Xn_train[13],
              Xn_train[14], Xn_train[15], Xn_train[16], Xn_train[17], Xn_train[18], Xn_train[19], Xn_train[20],
              Xn_train[21], Xn_train[22], Xn_train[23], Xn_train[24], Xn_train[25], Xn_train[26], Xn_train[27]], np.int32)
print(u)
plt.imshow(u, cmap='gray_r')
plt.show()

print("\nSome predictions")



# print("\nImage 1 is a five?\t ", y_train[0])
# plt.imshow(X_train[0], cmap='gray_r')
# plt.show()
# print("probability:\t\t\t ", model.predict(X_train[0].reshape(1,784)))
#
# print("\nImage 2 is a five?\t ", y_train[1])
# plt.imshow(X_train[1], cmap='gray_r')
# plt.show()
# print("probability:\t\t\t ", model.predict(X_train[1].reshape(1,784)))
#
# print("\nImage 645 is a five?\t ", y_train[644])
# plt.imshow(X_train[644], cmap='gray_r')
# plt.show()
# print("probability:\t\t\t ", model.predict(X_train[644].reshape(1,784)))
#
print("\nImage TEST is a five?\t ", u)
plt.imshow(u, cmap='gray_r')
plt.show()
print("probability:\t\t\t ", model.predict(u.reshape(1,784)))


for x in X_train:
    print(x)

