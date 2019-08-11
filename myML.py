from layers import Fully_Connected_Layer
import numpy as np 
import time
from helpers import flatten_img_list
import json
import pickle  
class NeuralNetwork():
  def __init__(self):
    self._layers = []
    self.losses = []

    self.training_data = None
    self.training_labels = None
  def save(self,filename):
    filehandler = open(filename, 'wb+') 
    pickle.dump(self, filehandler)
  def load(self,filename):
    filehandler = open(filename, 'rb') 
    return pickle.load(filehandler)
  def forward(self, X): 
    output = X
    for i in range(len(self._layers)):
      output = self._layers[i].forward(output)
    return output
  def __prep_data(self,X):
    if self._layers[0].type == "Dense":
      X = flatten_img_list(X)
    else:
      X = [[i] for i in X]
    return X
  def set_training_set(self,X,y):
    self.training_data = self.__prep_data(X)
    self.training_labels = y#self.__prep_data(y) 
  def add(self,layer):
    self._layers.append(layer)
    if len(self._layers) == 1:
      pass
    else:
      self._layers[-1]._prev = self._layers[-2]
      self._layers[-2]._next = self._layers[-1]
  def backward(self, y):
    for i in range(len(self._layers))[::-1]:
      self._layers[i].backward(y)
  # NOTE: These Calculations are too domain specific
  def get_acc(self,X,y):
    out = self.forward(self.__prep_data(X))
    right = sum([np.argmax(out[i]) == np.argmax(y[i]) for i in range(len(out))])
    return right/len(y)
  def get_recall(self,out):
    out = self.forward(self.training_data)
    a = sum([np.argmax(out[i]) == np.argmax(self.training_labels[i]) for i in range(len(out))])
    return a / len(self.training_data)
  def last_layer_size():
    pass
  def train (self,epochs = 1000, batch_size = 32, resolution = 10):
    start_time = time.time()
    for i in range(epochs):
      batch = np.random.choice(range(len(self.training_data)),batch_size)
      data = np.array([self.training_data[i] for i in batch])
      labels = np.array([self.training_labels[i] for i in batch])
      out = self.forward(data)
      self.backward(labels)
      if i % resolution == 0:
        self.losses.append(np.mean(np.square((labels - out))))
        print(f"@ Epoch: {i} of {epochs}\n\t Loss: {self.losses[-1]}")
    Seconds = int(time.time() - start_time)
    Minutes = Seconds // 60
    Seconds -= Minutes * 60

    Hours = Minutes // 60
    Minutes -= Hours * 60
    print(f"Total Training Time {Hours} Hours {Minutes} Minutes {Seconds} Seconds")
class AutoEncoder(NeuralNetwork):
  def __init__(self):
    NeuralNetwork.__init__(self)
    self.Bottleneck = None 
    pass
  def encode(self,X):
    if self.Bottleneck == None:
      self.Bottleneck = min(self._layers, key = lambda x: x.outputSize)
    encodings = []
    for data in X:
      layer = self.__layers[0]
      while layer != self.Bottleneck:
        data = layer.forward(data)
        layer = layer.next()
      data = self.Bottleneck.forward(data)
      encodings.append(data)
    return np.array(encodings)
  def decode(self,X):
    pass
  def sample(self,n = 10):
    if self.Bottleneck == None:
      self.Bottleneck = min(self._layers, key = lambda x: x.outputSize)
    samples = []
    for data in np.random.rand(n,self.Bottleneck.outputSize):
      layer = self.Bottleneck.next()
      while layer != None:
        data = layer.forward(data)
        layer = layer.next()
      samples.append(data)
    return np.array(samples)