from PIL import Image, ImageFilter
import numpy as np
import matplotlib.pyplot as plt



def convert_image(image):
    newImage = Image.new('L', (28, 28), (255))
    img = Image.open(image).convert('L').resize((28, 28), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
    newImage.paste(img, (0, 0))
    return [(255 - x) * 1.0 / 1 for x in list(newImage.getdata())]

def read_image(image):
    image_result = np.array_split(convert_image(image), 28)
    return np.array(image_result)



def show(img, figsize=(8, 4), title=None):
    plt.figure(figsize=figsize)
    plt.imshow(img, cmap='gray')
    if title:
        plt.title(title)
    plt.show()

test = read_image('image/five.png')

print(test)
show(test)