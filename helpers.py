import numpy as np
def ReLU(x):
    return x * (x > 0)
def ReLUPrime(x):
    return 1 * (x > 0)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoidPrime(s):
    return s * (1 - s)
def softMax(x):
  return np.exp(x)/sum(np.exp(x))
activations = {"relu":ReLU,"sigmoid":sigmoid,"tanh":np.tanh,"threshold":0,"softmax":None}
activations_prime = {"relu":ReLUPrime,"sigmoid":sigmoidPrime}

def flatten_img_list(x):
  x = np.array(x)

  return x.reshape((x.shape[0],x.shape[1] * x.shape[2]))
def flip(a):
    return np.fliplr(np.flipud(a))
