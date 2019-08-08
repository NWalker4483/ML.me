
import numpy as np
def ReLU(x):
    return x * (x > 0)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoidPrime(s):
    return s * (1 - s)
activations = {"relu":ReLU,"sigmoid":sigmoid,"tanh":np.tanh,"threshold":0}
activations_prime = {"sigmoid":sigmoidPrime}

class Fully_Connected_Layer():
  def __init__(self,input_size,layer_size,activation = "sigmoid"):
    self.weights = np.random.randn(input_size,layer_size)
    self.outputSize = layer_size
    self.next = None 
    self.prev = None 
    self.activation = activations[activation]
    self.activation_prime = activations_prime[activation]

    self.losses = []

  def forward(self,X):
    if self.prev == None: 
      self.activated_outputs = X
    else:
      self.outputs = np.dot(X, self.weights) 
      self.activated_outputs = self.activation(self.outputs)
    return self.activated_outputs
  def backward(self,y):
    if self.prev == None:
      return
    if self.next == None:
      self.error = y - self.activated_outputs # error in output
    else: 
      self.error = self.next.delta.dot(self.next.weights.T) 
    self.delta = self.error*self.activation_prime(self.activated_outputs)
    self.weights += self.prev.activated_outputs.T.dot(self.delta) * .001  
class Neural_Network():
  def __init__(self):
    self.__layers = []
    self.losses = []

    self.training_data = None
    self.training_labels = None

  def forward(self, X): 
    output = X
    for i in range(len(self.__layers)):
      output = self.__layers[i].forward(output)
    return output
  def set_training_set(self,X,y):
    self.training_data = X
    self.training_labels = y 
    # TODO: Seperate Input Layer Creation so that it may be a Conv Layer
    self.__layers.append(Fully_Connected_Layer(1,len(X[0])))

  def add_layer(self,size,activation="sigmoid"):
    self.__layers.append(Fully_Connected_Layer(self.__layers[-1].outputSize,size))
    self.__layers[-2].next = self.__layers[-1]
    self.__layers[-1].prev = self.__layers[-2]

  def backward(self, y):
    for i in range(len(self.__layers))[::-1]:
      self.__layers[i].backward(y)
  def get_acc(self,X,y):
    out = self.forward(X)
    right = sum([np.argmax(out[i]) == np.argmax(y[i]) for i in range(len(out))])
    return right/len(y)
  def get_recall(self):
    out = self.forward(self.training_data)
    a = sum([np.argmax(out[i]) == np.argmax(self.training_labels[i]) for i in range(len(out))])
    return a / len(self.training_data)

  def train (self,epochs = 1000, batch_size = 32, resolution = 200):
    for i in range(epochs):
      batch = np.random.choice(range(len(self.training_data)),batch_size)
      data = np.array([self.training_data[i] for i in batch])
      labels = np.array([self.training_labels[i] for i in batch])
      self.forward(data)
      self.backward(labels)
      if i % resolution == 0:
        out = self.forward(X)
        print(self.get_recall() * 100)
        self.losses.append(np.mean(np.square(y - out)))

from keras.datasets import mnist
import matplotlib.pyplot as plt

def flatten_img_list(x):
  return x.reshape((x.shape[0],x.shape[1] * x.shape[2]))
(x, y), (x2, y2) = mnist.load_data()
print("But not really!!!!")
X = flatten_img_list(x)
y = np.array([[i] for i in y])

#  map to 0 - 1 
X = X/255
dataset = np.array([np.zeros(10) for i in range(len(y))])
for i in range(len(y)):
  dataset[i][y[i]] = 1
    
net = Neural_Network()
net.set_training_set(X[:10000],dataset[:10000])
net.add_layer(6)
net.add_layer(10)
net.train(epochs = 9000,batch_size = 200, resolution = 5)
print(net.get_acc(X[10000:16500],dataset[10000:16500]) * 100)
plt.plot(range(len(net.losses)),net.losses)
plt.show()
