from helpers import activations, activations_prime
import numpy as np
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
    # TODO: Pass in learning rate from model  
    self.weights += self.prev.activated_outputs.T.dot(self.delta) * .001  
class ConvolutionalLayer():
    def __init__(self):
        
        pass