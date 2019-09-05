
import numpy as np
from myML import NeuralNetwork
from layers import ConvolutionalLayer, PoolingLayer, Dense, FlattenLayer
from helpers import flatten_img_list
import matplotlib.pyplot as plt
train = True
if train:
    print("Not Really...")
    from keras.datasets import mnist
    (x, y), (x2, y2) = mnist.load_data()
    y = np.array([[i] for i in y])
    #  map to 0 - 1 
    X = x/255
    # One Hot Encoding 
    dataset = np.array([np.zeros(10) for i in range(len(y))])
    for i in range(len(y)):
        dataset[i][y[i]] = 1
    X = [[X[i]] for i in range(len(X))]
    net = NeuralNetwork()
    net.add(ConvolutionalLayer(2,(3,3),stride=1,input_shape=(28,28))) # 6
    net.add(ConvolutionalLayer(3,(3,3),stride=1)) # 6
    #net.add(PoolingLayer((2,2),2))
    #net.add(ConvolutionalLayer(16,(3,3),stride=1))
    #net.add(PoolingLayer((2,2),2))
    net.add(FlattenLayer())
    #net.add(Dense(120))
    net.add(Dense(84))
    net.add(Dense(10))#,"softmax"))
    net.set_training_set(X[:2000],dataset[:2000])
    # NOTE: Only Increase epochs after successfully designed
    try:
        net.train(epochs = 50, batch_size = 45, resolution = 1)
    except KeyboardInterrupt as e:
        net.save("CNN.failed.model")
    finally:
        net.save("CNN.model")
else:
    net = NeuralNetwork().load("CNN.model") 
print(net.get_recall())
#print(net.get_acc(X[30000:31000],dataset[30000:31000]))
#plt.plot(range(len(net.losses)),net.losses)
#plt.title("Loss vs. Epochs")
#plt.show()
