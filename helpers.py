import numpy as np
def ReLU(x):
    return x * (x > 0)
def ReLUPrime(x):
    return 1 * (x > 0)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoidPrime(s):
    return s * (1 - s)
activations = {"relu":ReLU,"sigmoid":sigmoid,"tanh":np.tanh,"threshold":0}
activations_prime = {"relu":ReLUPrime,"sigmoid":sigmoidPrime}
def scan():
    k = (3,3)
    outs = []
    sample_img = np.zeros((20,20))
    offset = int(k[0]/2)
    x,y = offset,offset
    stride = 1

    x_ops = sample_img.shape[0] - k[0] + 1 // stride 
    y_ops = sample_img.shape[1] - k[1] + 1 // stride
    final = np.zeros((x_ops,y_ops))
    for _y in range(y_ops):
        for _x in range(x_ops):
            outs.append(sample_img[y-offset:y+offset+1][:,x-offset:x+offset+1])
            x+=stride 
        x = offset
        y+=stride
    try:
        assert(all([out.shape == k for out in outs]))
        print(len(outs))
    except AssertionError:
        print(*[i.shape for i in outs], sep="\n")

def flatten_img_list(x):
  x = np.array(x)
  return x.reshape((x.shape[0],x.shape[1] * x.shape[2]))
if __name__ == "__main__":
    print(MaxPooling(np.zeros((16,16)),stride = 4,filter_shape=(4,4)))