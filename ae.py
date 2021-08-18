import numpy as np
from ml_me.architectures import AutoEncoder
from ml_me.layers import Dense
from ml_me.helpers import flatten_img_list
import cv2
from keras.datasets import mnist
import matplotlib.pyplot as plt

#  map to 0 - 1 
(x1, y1), (x2, y2) = mnist.load_data()
print("But not really!!!!")

X = x1/255
print("Here its trying to generate 7's")
X = [X[i] for i in range(len(X)) if y1[i] == 7]
#X = flatten_img_list(X)

net = AutoEncoder()
net.add(Dense(784))
net.add(Dense(32))
net.add(Dense(8))
net.add(Dense(32))
net.add(Dense(784))

X = flatten_img_list(X[:30000])
net.set_training_set(X,X)
#net.encode(7)
try:
    net.train(epochs = 500,batch_size = 200)
except KeyboardInterrupt as e:
    net.save("AE.failed.model")
finally:
    net.save("AE.model")

for img in [i.reshape(28,28) for i in net.sample()]:
  img = cv2.resize(img, (300,300))
  cv2.imshow("",img)
  cv2.waitKey(0)
  cv2.imshow("",np.zeros_like(img))
  cv2.waitKey(0)
plt.plot(range(len(net.losses)),net.losses)
plt.title("Loss vs. Epochs")
plt.show()
