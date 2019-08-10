from helpers import activations, activations_prime, flatten_img_list
import numpy as np
class Layer():
    def __init__(self,Type,activation="sigmoid"):
        self.type = Type
        self._prev = None 
        self._next = None 
        self.activation = activations[activation]
        self.activation_prime = activations[activation]

    def prev(self):
        if self._prev != None:
            if self._prev.type == "Pooling":
                return self._prev.prev()
        return self._prev
    def next(self):
        if self._next != None:
            if self._next.type == "Pooling":
                return self._next.next()
        return self._next
    def __len__(self):
        return len(self.weights)
    
class Fully_Connected_Layer(Layer):
    def __init__(self,input_size,layer_size,activation = "sigmoid"):
        Layer.__init__(self, "Dense")
        self.bais = np.zeros(layer_size)
        self.weights = np.random.randn(input_size,layer_size)
        self.outputSize = layer_size
    def forward(self,X): 
        if self._prev == None: 
            self.outputs = X
        else:
            if self.prev().type in ["Conv","Pooling"]: #Bad
                X = self.prev().outputVector()
            self.unactivated_outputs = np.dot(X, self.weights) 
            self.outputs = self.activation(self.unactivated_outputs)
        return self.outputs
    def backward(self,y):
        if self._prev == None:
            return
        if self._next == None:
            self.error = y - self.outputs # error in output
        else: 
            self.error = self.next().delta.dot(self.next().weights.T) # Represents the direction in which the weights of the current layer need to change in order to correct the error of the next forward layer 
        self.delta = self.error*self.activation_prime(self.outputs)
        # TODO: Pass in learning rate from model  
        if self.prev().type == "Conv": #Bad
            self.weights += self.prev().outputVector().T.dot(self.delta) * .001  
        else:
            self.weights += self.prev().outputs.T.dot(self.delta) * .001  
    def description(self): # Provide String representation to store the model 
        pass
class PoolingLayer(Layer):
    def __init__(self,filter_shape=(2,2), stride=2, padding = False):
        Layer.__init__(self,"Pooling")
        self.stride = stride
        self.filter_shape = filter_shape
    def backward(self, y): # Bad
        pass
    def forward(self,_3Dvolumes): # Good 
        outputs = [] 
        for volume in _3Dvolumes:
            Pools = []
            for Slice in volume:
                Pools.append(self.MaxPooling(Slice))
            outputs.append(Pools)
        self.outputs = np.array(outputs)
        return self.outputs
    def MaxPooling(self,img): # Good // Parallelize
        k = self.filter_shape
        offset = int(k[0]/2)
        x,y = 0, 0  # offset,offset
        # NOTE: These op counts are unchecked 
        x_ops = ((img.shape[0] - k[0])// self.stride) + 1 
        y_ops = ((img.shape[1] - k[1])// self.stride) + 1
        final = np.zeros((x_ops,y_ops))
        for _y in range(y_ops):
            for _x in range(x_ops):
                final[_y][_x] = img[y:y+k[0]][:,x:x+k[1]].max()
                x += self.stride
            x = 0 
            y += self.stride
        return final
    def description(self): # Provide String representation to store the model 
        return f"{self.type} {self.filter_shape} {self.stride}"
class ConvolutionalLayer(Layer):
    def __init__(self, layer_size, filter_shape = (3,3) , stride=1, padding = False,activation = "relu"):
        Layer.__init__(self,"Conv",activation)
        self.weights = np.random.randn(layer_size,np.product(filter_shape))
        self.filter_shape = filter_shape
        self.stride = stride
    def forward(self,_3Dvolumes):
        self.outputs = []
        for volume in _3Dvolumes:
            Convolutions = []
            for Slice in volume:
                for Filter in self.weights: # NOTE: Is equivilant to querying each individual perceptron
                    Filter = Filter.reshape(self.filter_shape)
                    Convolutions.append(self.activation(self.Convolve(Slice,Filter)))
            self.outputs.append(Convolutions)
        self.outputs = np.array(self.outputs)
        return self.outputs
    def outputVector(self):
        outs = []
        for volume in self.outputs:
            outs.append(flatten_img_list(volume).flatten())
        return np.array(outs)
    def backward(self,y): # Bad
        if self.prev() == None:
            return
        if self.next() == None:
            self.error = y - self.outputs # error in output
        else: 
            self.error = self.next().delta.dot(self.next().weights.T) # Represents the direction in which the weights of the current layer need to change in order to correct the error of the next forward layer 
        self.delta = self.error*self.activation_prime(self.outputVector())
        # TODO: Pass in learning rate from model  
        if self.prev().type == "Conv":
            self.weights += self.prev().outputVector().T.dot(self.delta) * .001  
        else:
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
                final[_y][_x] = sum(sum(_filter * img[y-offset:y+offset+1][:,x-offset:x+offset+1]))
                x+=self.stride
            x = offset
            y+=self.stride
        return final
    def description(self): # Provide String representation to store the model 
        pass