from PIL import Image, ImageFilter
import numpy as np
import matplotlib.pyplot as plt

class convert_image(object):
    def __init__(self, image:object):
        self.image = image

def imageprepare(image):
    """
    This function returns the pixel values.
    The imput is a png file location.
    """
    im = Image.open(image).convert('L')
    width = float(im.size[0])
    height = float(im.size[1])
    newImage = Image.new('L', (28, 28), (255))  # creates white canvas of 28x28 pixels

    if width > height:  # check which dimension is bigger
        # Width is bigger. Width becomes 20 pixels.
        nheight = int(round((28.0 / width * height), 0))  # resize height according to ratio width
        if (nheight == 0):  # rare case but minimum is 1 pixel
            nheight = 1
            # resize and sharpen
        img = im.resize((28, nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((28 - nheight) / 2), 0))  # calculate horizontal position
        newImage.paste(img, (0, wtop))  # paste resized image on white canvas
    else:
        # Height is bigger. Heigth becomes 20 pixels.
        nwidth = int(round((28.0 / height * width), 0))  # resize width according to ratio height
        if (nwidth == 0):  # rare case but minimum is 1 pixel
            nwidth = 1
            # resize and sharpen
        img = im.resize((nwidth, 28), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((28 - nwidth) / 2), 0))  # caculate vertical pozition
        newImage.paste(img, (wleft, 0))  # paste resized image on white canvas

    # newImage.save("sample.png

    tv = list(newImage.getdata())  # get pixel values

    # normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
    tva = [(255 - x) * 1.0 / 255.0 for x in tv]
    return tva

def readImage(img):
    x_1 = imageprepare(img)
    y_1 = np.array_split(x_1, 28)
    return np.array(y_1)

def show(img, figsize=(8, 4), title=None):
    plt.figure(figsize=figsize)
    plt.imshow(img, cmap='gray')
    if title:
        plt.title(title)
    plt.show()

z_0 = readImage('image/null.png')
z_1 = readImage('image/one.png')
z_2 = readImage('image/two.png')
z_3 = readImage('image/three.png')
z_4 = readImage('image/four.png')
z_5 = readImage('image/five.png')
z_6 = readImage('image/six.png')
z_7 = readImage('image/seven.png')
z_8 = readImage('image/eight.png')
z_9 = readImage('image/nine.png')
test = readImage('image/test.png')

print(test)
show(test)