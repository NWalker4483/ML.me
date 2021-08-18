import numpy as np
from ml_me.architectures import NeuralNetwork
from ml_me.layers import Dense
from ml_me.helpers import flatten_img_list

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

#X = [X[i] for i in range(len(X)) if y[i]]
X = flatten_img_list(X)
net = NeuralNetwork()
net.add(Dense(784))
net.add(Dense(16))
net.add(Dense(16))
net.add(Dense(10))

net.set_training_set(X[:3000],Y[:3000])
net.train(epochs = 10000, batch_size = 200)
net.save("MLP.model")
print(f"Recall: {net.get_recall()} Accuracy: {round(net.get_acc(X, Y), 3)}")
plt.plot(range(len(net.losses)),net.losses)
plt.title("Loss vs. Epochs")
plt.show()
