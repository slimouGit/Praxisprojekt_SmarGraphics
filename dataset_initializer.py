import config
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

image = Image.open(config.filename)

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

bw_image = image.convert(mode='L')
show(bw_image, figsize=(11, 11))
bw_image = ImageEnhance.Contrast(bw_image).enhance(1.5)
show(bw_image, figsize=(11, 11))

#--------------------------------------------------------------

SIZE = 28
samples = []
for digit, y in enumerate(range(0, bw_image.height, SIZE)):
    print('Cutting digit:', digit)
    cuts=[]
    for x in range(0, bw_image.width, SIZE):
        cut = bw_image.crop(box=(x, y, x+SIZE, y+SIZE))
        cuts.append(cut)
    samples.append(cuts)
print(f'Cut {len(samples)*len(samples[0])} images total.')

#--------------------------------------------------------------

f = plt.figure(figsize=(18,2))
ax = f.subplots(1, 6)
for i, n in enumerate(np.random.randint(0, len(samples), size=6)):
    m = (np.random.randint(0, len(samples[n])))
    ax[i].imshow(samples[n][m])
    ax[i].set_title(f'Digit: [{n}]')
plt.show()

#--------------------------------------------------------------

sample = samples[7][14]
show(sample, figsize=(2, 2))

#--------------------------------------------------------------

from PIL import ImageOps
import matplotlib.patches as patches

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

crop = inv_sample.crop(bbox)
show(crop, title='Image cropped to bounding box')

new_size = 28
delta_w = new_size - crop.size[0]
delta_h = new_size - crop.size[1]
padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
new_im = ImageOps.expand(crop, padding)
show(new_im, title='Resized and centered to 28x28')

#--------------------------------------------------------------

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

f = plt.figure(figsize=(18,2))
ax = f.subplots(1, 6)
for i, n in enumerate(np.random.randint(0, len(resized_samples), size=6)):
    m = (np.random.randint(0, len(resized_samples[n])))
    ax[i].imshow(resized_samples[n][m])
    ax[i].set_title(f'Digit: [{n}]')
plt.show()

#--------------------------------------------------------------

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
preview.save('image/preview.png')

#--------------------------------------------------------------

binary_samples = np.array([[sample.getdata() for sample in row] for row in resized_samples])
binary_samples = binary_samples.reshape(len(resized_samples)*len(resized_samples[0]), 28, 28)

classes = np.array([[i]*config.columns for i in range(10)]).reshape(-1)
print(f'X shape: {binary_samples.shape}')
print(f'y shape: {classes.shape}')

np.save(config.xfile, binary_samples)
np.save(config.yfile, classes)

