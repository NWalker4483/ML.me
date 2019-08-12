
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
Y = np.array([np.zeros(10) for i in range(len(y))])
for i in range(len(y)):
  Y[i][y[i]] = 1
X = [X[i] for i in range(len(X)) if y[i]]

net = NeuralNetwork()
net.add(Fully_Connected_Layer(0,784))
net.add(Fully_Connected_Layer(784,16))
net.add(Fully_Connected_Layer(16,16))
net.add(Fully_Connected_Layer(16,10))

net.set_training_set(X[:3000],Y[:3000])
net.train(epochs = 1000,batch_size = 200)
net.save("MLP.model")

plt.plot(range(len(net.losses)),net.losses)
plt.title("Loss vs. Epochs")
plt.show()
