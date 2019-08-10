from layers import Fully_Connected_Layer
import numpy as np 
class NeuralNetwork():
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
    self.__layers[-2]._next = self.__layers[-1]
    self.__layers[-1]._prev = self.__layers[-2]
  def add(self,layer):
    self.__layers.append(layer)
    if len(self.__layers) == 1:
      pass
    else:
      self.__layers[-1]._prev = self.__layers[-2]
      self.__layers[-2]._next = self.__layers[-1]
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
        out = self.forward(data)
        print(self.get_recall() * 100)
        self.losses.append(np.mean(np.square(labels - out)))
