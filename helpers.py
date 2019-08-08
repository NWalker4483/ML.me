import numpy as np
def ReLU(x):
    return x * (x > 0)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoidPrime(s):
    return s * (1 - s)
activations = {"relu":ReLU,"sigmoid":sigmoid,"tanh":np.tanh,"threshold":0}
activations_prime = {"sigmoid":sigmoidPrime}

def flatten_img_list(x):
  return x.reshape((x.shape[0],x.shape[1] * x.shape[2]))
