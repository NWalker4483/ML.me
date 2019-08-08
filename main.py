
import numpy as np
from myML import NeuralNetwork
from helpers import flatten_img_list
from keras.datasets import mnist
import matplotlib.pyplot as plt

#  map to 0 - 1 
(x, y), (x2, y2) = mnist.load_data()
print("But not really!!!!")
X = flatten_img_list(x)
y = np.array([[i] for i in y])

X = X/255
dataset = np.array([np.zeros(10) for i in range(len(y))])
for i in range(len(y)):
  dataset[i][y[i]] = 1
    
net = NeuralNetwork()
net.set_training_set(X[:10000],dataset[:10000])
net.add_layer(16)
net.add_layer(16)
net.add_layer(10)
net.train(epochs = 9000,batch_size = 200, resolution = 5)
print(net.get_acc(X[10000:16500],dataset[10000:16500]) * 100)
plt.plot(range(len(net.losses)),net.losses)
plt.show()
