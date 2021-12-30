from createModel import PredictiveModel
import matplotlib.pyplot as plt

PredictiveModel(21, 5000, 1)

def predictAndShow(img):
    plt.imshow(img)
    plt.show()
    print("probability of:\t\t\t ", model.predict(img.reshape(1, 784)))
    if ((model.predict(img.reshape(1, 784))) > (0.6)):
        print("TRUE")
    else:
        print("FALSE")

predictAndShow(z_0)
