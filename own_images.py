from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter

from convert_image import convert_image

input_folder = 'image/'

columns = 1344
epoche = 10
hidden_layer = 100
reference = 0
border = 6.0
batch = 10
filename = input_folder+'numbers1344.jpg'
image = Image.open(filename)
predictedValues = []
images = []

def show(img, figsize=(8, 4), title=None):
    plt.figure(figsize=figsize)
    plt.imshow(img)
    if title:
        plt.title(title)
    plt.show()

show(image)

#--------------------------------------------------------------

print(f'Image details')
print(f'Format {image.format}')
print(f'Size {image.size}')
print(f'Mode {image.mode}')

#--------------------------------------------------------------

from PIL import ImageEnhance

bw_image = image.convert(mode='L') #L is 8-bit black-and-white image mode
# bw_image = image
show(bw_image, figsize=(12, 12))
bw_image = ImageEnhance.Contrast(bw_image).enhance(1.5)
show(bw_image, figsize=(12, 12))

#--------------------------------------------------------------

#Cut square images 30x30 pixels each.

SIZE = 28
samples = [] #array to store cut images
for digit, y in enumerate(range(0, bw_image.height, SIZE)):
    #print('Cutting digit:', digit)
    cuts=[]
    for x in range(0, bw_image.width, SIZE):
        cut = bw_image.crop(box=(x, y, x+SIZE, y+SIZE))
        cuts.append(cut)
    samples.append(cuts)
print(f'Cut {len(samples)*len(samples[0])} images total.')

#--------------------------------------------------------------

#Now samples contain 10 arrays, each for its digit.
#Let's have a look at several random images:

f = plt.figure(figsize=(18,2))
ax = f.subplots(1, 6)
for i, n in enumerate(np.random.randint(0, len(samples), size=6)):
    m = (np.random.randint(0, len(samples[n])))
    ax[i].imshow(samples[n][m])
    ax[i].set_title(f'Digit: [{n}]')
plt.show()

#--------------------------------------------------------------

#Center images

sample = samples[7][14]
show(sample, figsize=(2, 2))

#In order to find and fix its location,
# I have to invert pixels, so that background area becomes black,
# and filled with zero pixels.
# Then getbbox method gives us the digit location, which has to be centered.

#--------------------------------------------------------------

from PIL import ImageOps
import matplotlib.patches as patches

# Inver sample, get bbox and display all that stuff.
inv_sample = ImageOps.invert(sample)
bbox = inv_sample.getbbox()

fig = plt.figure(figsize=(2, 2))
ax = fig.add_axes([0,0,1,1])

ax.imshow(inv_sample)
rect = patches.Rectangle(
    (bbox[0], bbox[3]), bbox[2]-bbox[0], -bbox[3]+bbox[1]-1,
    fill=False, alpha=1, edgecolor='w')
ax.add_patch(rect)
plt.show()

#--------------------------------------------------------------

#And here is how to center image and resize it to desired size again:
crop = inv_sample.crop(bbox)
show(crop, title='Image cropped to bounding box')

#resize back
new_size = 28
delta_w = new_size - crop.size[0]
delta_h = new_size - crop.size[1]
padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
new_im = ImageOps.expand(crop, padding)
show(new_im, title='Resized and centered to 28x28')

#--------------------------------------------------------------

#And now I'm ready to put all those findings into a single function and resize all images:
def resize_and_center(sample, new_size=28):
    inv_sample = ImageOps.invert(sample)
    bbox = inv_sample.getbbox()
    crop = inv_sample.crop(bbox)
    delta_w = new_size - crop.size[0]
    delta_h = new_size - crop.size[1]
    padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
    return ImageOps.expand(crop, padding)

resized_samples = []
for row in samples:
    resized_samples.append([resize_and_center(sample) for sample in row])

#--------------------------------------------------------------

#Let's have a look:

f = plt.figure(figsize=(18,2))
ax = f.subplots(1, 6)
for i, n in enumerate(np.random.randint(0, len(resized_samples), size=6)):
    m = (np.random.randint(0, len(resized_samples[n])))
    ax[i].imshow(resized_samples[n][m])
    ax[i].set_title(f'Digit: [{n}]')
plt.show()

#--------------------------------------------------------------

#Well, I think the digits are cleaned up and nicely centered.
# Some of them look a little bit overrown, but that's ok with me.
# Let's glue all of the small images together and get a big picture.

preview = Image.new('L', (len(samples[0])*new_size, len(samples)*new_size))


x = 0
y = 0
for row in resized_samples:
    for sample in row:
        preview.paste(sample, (x, y))
        x += new_size
    y+=new_size
    x = 0

show(preview, figsize=(18,18), title='Processed images')
preview.save('preview.png')

#--------------------------------------------------------------

#Save the result in numpy binary format¶
#method returns image bytes, which we put into numpy array.
#Image.getdata()

binary_samples = np.array([[sample.getdata() for sample in row] for row in resized_samples])
binary_samples = binary_samples.reshape(len(resized_samples)*len(resized_samples[0]), 28, 28)

#As we had 10 columns and 10 rows in the original picture,
# now we generate a target array with corresponging digit

classes = np.array([[i]*columns for i in range(10)]).reshape(-1)
print(f'X shape: {binary_samples.shape}')
print(f'y shape: {classes.shape}')

#Save files to numpy binary format...
xfile = 'digits_x_test.npy'
yfile = 'digits_y_test.npy'
np.save(xfile, binary_samples)
np.save(yfile, classes)

#...and test it
x_test = np.load(xfile)
y_test = np.load(yfile)
x_test.shape, y_test.shape

# for i in np.random.randint(x_test.shape[0], size=6):
#     show(x_test[i], title=f'Digit [{y_test[i]}]', figsize=(1,1))

#--------------------------------------------------------------

# show(x_test[0], title=f'Digit [{y_test[0]}]', figsize=(8,4))
# print(type(x_test[0]))
# print(x_test[0])
# print("lable ", y_test[0])
#
# show(x_test[22], title=f'Digit [{y_test[22]}]', figsize=(8,4))
# print(type(x_test[22]))
# print(x_test[22])
# print("lable ", y_test[22])

#--------------------------------------------------------------

# show(x_test[22])

#--------------------------------------------------------------

def bestimmeReferenzNummer(i):
    global y_test
    y_test = y_test == i  # Vergleich ueber alle Elemente im Daten-Array, Aufloesung nach T-Shirt == 0

bestimmeReferenzNummer(reference)

#--------------------------------------------------------------

from keras.models import Sequential
from keras.layers import Dense


def initializeModel(i):
    global model
    model = Sequential()
    # global y_test
    # y_test = y_test == i
    # hiddenlayer
    model.add(Dense(hidden_layer, activation="sigmoid", input_shape=(784,)))
    # outputlayer
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer="sgd", loss="binary_crossentropy")
    a = x_test.reshape(columns * 10, 784)
    model.fit(a, y_test, epochs=epoche, batch_size=batch)


initializeModel(0)


#--------------------------------------------------------------

def predictAndShow(expected, img):
    plt.imshow(img)
    plt.show()
    print("probability of predicting", reference ," by image with number ", expected, " is \t\t\t ", model.predict(img.reshape(1, 784)))
    if( model.predict(img.reshape(1, 784))>border):
        print("POSSIBLE")
    predictedValues.append(model.predict(img.reshape(1, 784)))

#--------------------------------------------------------------
possibleMatches = []

# def predictAndShow(img):
#     plt.imshow(img)
#     plt.show()
#     print("probability of predicting", reference , "is \t\t\t ", model.predict(img.reshape(1, 784)))
#     if( model.predict(img.reshape(1, 784))>0.3):
#         print("POSSIBLE")
#         possibleMatches.append(model.predict(img.reshape(1, 784)))

#--------------------------------------------------------------

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

#--------------------------------------------------------------

def readImage(img):
    x_1 = imageprepare(img)
    y_1 = np.array_split(x_1, 28)
    return np.array(y_1)

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

#--------------------------------------------------------------

def determinePredictedNumber(predictedValues):
    max_value = max(predictedValues)
    max_index = predictedValues.index(max_value)
    # print("predicted number ", reference , " could be at index number: ", max_index)
    # print("propability is ", max_value)
    if(max_value<(border*0.1)):
        print("It seems, searched number ", reference, "is not found")
    else:
        print("probably searched number ", reference, " was found at index ", max_index)


#--------------------------------------------------------------

def determinePossibleMatches(possibleMatches):
    if (len(possibleMatches) == 0):
        print("looks like, no number is ", reference)
    if (len(possibleMatches) == 1):
        print("index with high probability to be number is", possibleMatches)
    if(len(possibleMatches) > 1):
        max_value = max(possibleMatches)
        max_index = possibleMatches.index(max_value)
        print("index with high probability to be number ", reference, " is ", max_index)

#--------------------------------------------------------------

images.append(z_3)
images.append(z_8)
images.append(z_3)
images.append(z_0)
images.append(z_1)
images.append(z_6)
images.append(z_9)

# for x in images:
#     predictAndShow(x)

#--------------------------------------------------------------

print("-------------------- IMAGES --------------------")
print("REFERENCE NUMBER ", reference)
predictAndShow(0, z_0)
predictAndShow(1, z_1)
predictAndShow(2, z_2)
# predictAndShow(3, z_3)
predictAndShow(4, z_4)
predictAndShow(5, z_5)
predictAndShow(6, z_6)
# predictAndShow(7, z_7)
predictAndShow(8, z_8)
# predictAndShow(9, z_9)


# predictAndShow( z_0)
# predictAndShow( z_1)
# predictAndShow( z_2)
# predictAndShow( z_3)
# predictAndShow( z_4)
# predictAndShow( z_5)
# predictAndShow( z_6)
# predictAndShow( z_7)
# predictAndShow( z_8)
# predictAndShow( z_9)

determinePredictedNumber(predictedValues)

# determinePossibleMatches(possibleMatches)



#--------------------------------------------------------------

