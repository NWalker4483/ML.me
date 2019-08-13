import numpy as np
from myML import NeuralNetwork ,AutoEncoder
from layers import ConvolutionalLayer, PoolingLayer, Dense
from keras.datasets import mnist
import matplotlib.pyplot as plt
from helpers import flatten_img_list

#  map to 0 - 1 
(x1, y1), (x2, y2) = mnist.load_data()
print("But not really!!!!")

X = x1/255
X = [X[i] for i in range(len(X)) if y1[i] == 7]

net = AutoEncoder()
net.add(Dense(0,784))
net.add(Dense(784,32))
net.add(Dense(32,8))
net.add(Dense(8,32))
net.add(Dense(32,784))

X = flatten_img_list(X[:10000])
net.set_training_set(X,X)
#net.encode(7)
try:
    net.train(epochs = 5000,batch_size = 200)
except KeyboardInterrupt as e:
    net.save("ae.failed.model")
finally:
    net.save("ae.model")
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
