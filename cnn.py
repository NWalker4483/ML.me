
import numpy as np
from myML import NeuralNetwork
from layers import ConvolutionalLayer, PoolingLayer, Fully_Connected_Layer
from helpers import flatten_img_list

import matplotlib.pyplot as plt

print("Not Really...")
from keras.datasets import mnist
(x, y), (x2, y2) = mnist.load_data()

y = np.array([[i] for i in y])
#  map to 0 - 1 
X = x/255
# One Hot Encoding 
dataset = np.array([np.zeros(10) for i in range(len(y))])
for i in range(len(y)):
  dataset[i][y[i]] = 1
X = [X[i] for i in range(len(X)) if y[i] == 7]

net = NeuralNetwork()
net.add(ConvolutionalLayer(1,(5,5),stride=1))
#net.add(PoolingLayer((2,2),2))
net.add(ConvolutionalLayer(2,(3,3),stride=1))
#net.add(PoolingLayer((2,2),2))

# TransformingLayer()

net.add(Fully_Connected_Layer(11616,120)),
net.add(Fully_Connected_Layer(120,84))
net.add(Fully_Connected_Layer(84,10))

net.set_training_set(X[:3000],X[:3000])
# NOTE: Only Increase epochs after successfully designed
net.train(epochs = 10, batch_size = 200)

net.save("CNN.model")

plt.plot(range(len(net.losses)),net.losses)
plt.title("Loss vs. Epochs")
plt.show()
