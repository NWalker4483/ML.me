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
    def prev(self):
        if self.prev != None:
            if self.prev.type == "Pooling":
                return self.prev.prev
        return self.prev
    def next(self):
        if self.next != None:
            if self.next.type == "Pooling":
                return self.next.next
        return self.next
    def forward(self,X):
        if self.prev == None: 
            self.outputs = X
        else:
            self.unactivated_outputs = np.dot(X, self.weights) 
            self.outputs = self.activation(self.unactivated_outputs)
        return self.outputs
    def backward(self,y):
        if self.prev == None:
            return
        if self.next == None:
            self.error = y - self.outputs # error in output
        else: 
            self.error = self.next().delta.dot(self.next().weights.T) # Represents the direction in which the weights of the current layer need to change in order to correct the error of the next forward layer 
        self.delta = self.error*self.activation_prime(self.outputs)
        # TODO: Pass in learning rate from model  
        self.weights += self.prev().outputs.T.dot(self.delta) * .001  
    def description(self): # Provide String representation to store the model 
        pass
class PoolingLayer():
    def __init__(self,filter_shape=(2,2), stride=2, padding = False):
        self.next = None
        self.prev = None
        self.stride = stride
        self.type = "Pooling"
        pass
    def forward(self,_3Dvolumes):
        self.outputs = [] 
        for volume in _3Dvolumes:
            Pools = []
            for Slice in volume:
                Pools.append(self.MaxPooling(Slice))
            self.outputs.append(Pools)
    def MaxPooling(self,img):
        k = self.filter_shape
        offset = int(k[0]/2)
        x,y = offset,offset
        # NOTE: These op counts are unchecked 
        x_ops = (img.shape[0] - k[0] // self.stride) + 1 
        y_ops = (img.shape[1] - k[1] // self.stride) + 1
        final = np.zeros((x_ops,y_ops))
        for _y in range(y_ops):
            for _x in range(x_ops):
                final[_y][_x] = img[y-offset:y+offset+1][:,x-offset:x+offset+1].max()
                x+=self.stride
            x = offset
            y+=self.stride
        return final
    def AvgPooling(self,img):
        k = self.filter_shape
        offset = int(k[0]/2)
        x,y = offset,offset
        # NOTE: These op counts are unchecked 
        x_ops = (img.shape[0] - k[0] // self.stride) + 1 
        y_ops = (img.shape[1] - k[1] // self.stride) + 1
        final = np.zeros((x_ops,y_ops))
        for _y in range(y_ops):
            for _x in range(x_ops):
                final[_y][_x] = np.average(img[y-offset:y+offset+1][:,x-offset:x+offset+1])
                x+=self.stride
            x = offset
            y+=self.stride
        return final
    def description(self): # Provide String representation to store the model 
        pass 
class ConvolutionalLayer():
    def __init__(self, layer_size, filter_shape = (3,3) , stride=1, padding = False,activation = "relu"):
        self.__filters = np.random.randn(layer_size,np.product(filter_shape))
        self.filter_shape = filter_shape
        self.activation = activations[activation]
        self.next = None 
        self.prev = None
        self.stride = stride
        self.type = "Conv"
    def prev(self):
        if self.prev != None:
            if self.prev.type == "Pooling":
                return self.prev.prev
        return self.prev
    def next(self):
        if self.next != None:
            if self.next.type == "Pooling":
                return self.next.next
        return self.next
    def forward(self,_3Dvolumes):
        self.outputs = []
        for volume in _3Dvolumes:
            Convolutions = []
            for Slice in volume:
                for Filter in self.__filters: # NOTE: Is equivilant to querying each individual perceptron
                    Filter = Filter.reshape(self.filter_shape)
                    Convolutions.append(self.activation(self.Convolve(Slice,Filter)))
            self.outputs.append(Convolutions)
    def backward(self,y):
        if self.prev == None:
            return
        if self.next == None:
            self.error = y - self.outputs # error in output
        else: 
            self.error = self.next().delta.dot(self.next().weights.T) # Represents the direction in which the weights of the current layer need to change in order to correct the error of the next forward layer 
        self.delta = self.error*self.activation_prime(self.outputs)
        # TODO: Pass in learning rate from model  
        self.weights += self.prev().outputs.T.dot(self.delta) * .001  
    def Convolve(self,img,_filter):
        k = _filter.shape
        offset = int(k[0]/2)
        x,y = offset,offset
        # NOTE: These op counts are unchecked 
        x_ops = (img.shape[0] - k[0] // self.stride) + 1 
        y_ops = (img.shape[1] - k[1] // self.stride) + 1
        final = np.zeros((x_ops,y_ops))
        for _y in range(y_ops):
            for _x in range(x_ops):
                final[_y][_x] = _filter.dot(img[y-offset:y+offset+1][:,x-offset:x+offset+1])
                x+=self.stride
            x = offset
            y+=self.stride
        return final
    def description(self): # Provide String representation to store the model 
        pass