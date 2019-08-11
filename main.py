
import numpy as np
from myML import NeuralNetwork ,AutoEncoder
from layers import ConvolutionalLayer, PoolingLayer, Fully_Connected_Layer
from helpers import flatten_img_list
from keras.datasets import mnist
import matplotlib.pyplot as plt

#  map to 0 - 1 
(x, y), (x2, y2) = mnist.load_data()
print("But not really!!!!")

y = np.array([[i] for i in y])

X = x/255
# One Hot Encoding 
dataset = np.array([np.zeros(10) for i in range(len(y))])
for i in range(len(y)):
  dataset[i][y[i]] = 1
X = [X[i] for i in range(len(X)) if y[i] == 7]
net = AutoEncoder()
net.add(Fully_Connected_Layer(0,784))
net.add(Fully_Connected_Layer(784,16))
net.add(Fully_Connected_Layer(16,784))

net.set_training_set(X[:3000],X[:3000])
#net.encode(7)
net.train(epochs = 1,batch_size = 200)
net.save("myML.model")
net.load("myML.model")

import cv2
for img in [i.reshape(28,28) for i in net.sample()]:
  img = cv2.resize(img, (300,300))
  cv2.imshow("",img)
  cv2.waitKey(0)
  cv2.imshow("",np.zeros_like(img))
  cv2.waitKey(0)
plt.plot(range(len(net.losses)),net.losses)
plt.title("Loss vs. Epochs")
plt.show()
