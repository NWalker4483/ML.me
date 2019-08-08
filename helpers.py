import numpy as np
def ReLU(x):
    return x * (x > 0)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoidPrime(s):
    return s * (1 - s)
activations = {"relu":ReLU,"sigmoid":sigmoid,"tanh":np.tanh,"threshold":0}
activations_prime = {"sigmoid":sigmoidPrime}
def MaxPooling(x,filter_shape=(3,3),padding = False):
    # TODO: Import Code from my Mac
    pass
def Convolve(x,filters=[],padding=False):
    # TODO: Import Code from my Mac
    pass
def flatten_img_list(x):
  return x.reshape((x.shape[0],x.shape[1] * x.shape[2]))
if __name__ == "__main__":
     def updateConfidence(self,pos, mod=.2, k_size=3):
        if k_size%2 == 0:
            print("Bad Kernel Size")
            return None
        x_lim = 8 #len(board[0])
        y_lim = 8 #len(board)
        offset = k_size//2
        for x in range(pos[0] - offset, pos[0] + offset + 1):
            for y in range(pos[1] - offset, pos[1] + offset + 1):
                if (x < x_lim and y < y_lim):
                    self.zone_confidence[y][x] = self.zone_confidence[y][x] + mod
                    if self.zone_confidence[y][x] < 0: 
                        self.zone_confidence[y][x] = 0 
                    elif self.zone_confidence[y][x] > 1: 
                        self.zone_confidence[y][x] = 1 