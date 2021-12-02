import tensorflow as tf
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

import matplotlib.pyplot as plt

image_index = 7777 # You may select anything up to 60,000
print("Number is ", y_train[image_index]) # The label is 8
plt.imshow(x_train[image_index], cmap='Greys')
plt.show()


# for y in y_train:
    # print(y_train[y])
    # print(type(y_train[y]))

index = 0
for y in y_train:
    print(y_train[y])


import numpy as np

# train_filter = np.where((y_train == 0 ))
# test_filter = np.where((y_test == 0))
#
# X_train, Y_train = x_train[train_filter], y_train[train_filter]
# X_test, Y_test = x_test[test_filter], y_test[test_filter]
#
# train_mask = np.isin(Y_train, [0])
# test_mask = np.isin(Y_test, [0])
#
# print(train_mask)
# print(test_mask)



