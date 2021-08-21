
import numpy as np
import ml_me.helpers as help
import json
class Layer():
    def __init__(self, Type, activation="sigmoid", **kwargs):
        self.type = Type
        self._prev = None 
        self._next = None 
        self.activation = help.activations[activation]
        self.activation_prime = help.activations[activation]

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

class Dense(Layer):
    def __init__(self, layer_size, **kwargs):
        Layer.__init__(self, "Dense", **kwargs)
        self.bias = np.zeros(layer_size)
        self.outputSize = layer_size

    def init(self,input_size):
        self.weights = np.random.randn(input_size, self.outputSize)

    def forward(self,X): 
        if self._prev == None: 
            self.outputs = X
        else:
            self.unactivated_outputs = np.dot(X, self.weights) 
            self.unactivated_outputs += self.bias
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
        self.weights += self.prev().outputs.T.dot(self.delta) * .001  
    def description(self): # Provide String representation to store the model 
        desc = {}
        desc["type"] = self.type
        desc["size"] = self.outputSize
        desc["weights"] = json.dumps(self.weights.tolist())
        return desc

class FlattenLayer(Layer):
    def __init__(self):
        Layer.__init__(self,"Flattening")
        pass
    def init(self,X):
        self.outputSize = np.prod(X)
        self.img_shape = None 
    def forward(self,x):
        #Reshape images from 
        self.outputs = x.reshape((x.shape[0],x.shape[1] * x.shape[2] * x.shape[3]))
        return self.outputs
        
    def backward(self,y):
        self.error = self.next().delta.dot(self.next().weights.T) # Represents the direction in which the weights of the current layer need to change in order to correct the error of the next forward layer 
        self.delta = self.error*self.activation_prime(self.outputs)
        self.delta = self.delta.reshape((self.delta.shape[0],self.prev().outputSize[0],self.prev().outputSize[1],self.prev().outputSize[2]))
        
        # Reshapes Deltas of Weights
    
class PoolingLayer(Layer): # Should act as a preserver of weights MAYBE NOT THOUGH 
    def __init__(self,filter_shape=(2,2), stride=2, padding = False):
        Layer.__init__(self,"Pooling")
        self.stride = stride
        self.filter_shape = filter_shape
        self.max_maps = None
        self.deltas = None 
    def backward(self, y): # Bad
        pass
    def forward(self,_3Dvolumes): # Good 
        outputs = [] 
        for volume in _3Dvolumes:
            Pools = []
            for Slice in volume:
                Pools.append(self.MaxPooling(Slice,True))
            outputs.append(Pools)
        self.outputs = np.array(outputs)
        return self.outputs
    def MaxPooling(self,img,Map = False): # Good // Parallelize
        k = self.filter_shape
        offset = int(k[0]/2)
        x,y = 0, 0  # offset,offset
        # NOTE: These op counts are unchecked 
        x_ops = ((img.shape[0] - k[0])// self.stride) + 1 
        y_ops = ((img.shape[1] - k[1])// self.stride) + 1
        final = np.zeros((x_ops,y_ops))
        self.max_map = np.zeros((x_ops,y_ops))
        for _y in range(y_ops):
            for _x in range(x_ops):
                pnt = np.argmax(img[y:y+k[0]][:,x:x+k[1]])
                Map[_y][_x] = (y + (pnt // k[0]), x + (pnt % 3))
                final[_y][_x] = 0
                x += self.stride
            x = 0 
            y += self.stride
        return final
    def description(self): # Provide String representation to store the model 
        pass#return f"{self.type} {self.filter_shape} {self.stride}"
class ConvolutionalLayer(Layer):
    def __init__(self, layer_size, filter_shape = (3,3) , stride=1, padding = False, activation = "relu", input_shape=(-1,-1)):
        Layer.__init__(self,"Conv", activation)
        self.bias = None 
        self.weights = np.random.randn(layer_size,np.product(filter_shape))
        self.filter_shape = filter_shape
        self.stride = stride
        self.layer_size = layer_size
        x_ops = (input_shape[0] - filter_shape[0] // self.stride) + 1 
        y_ops = (input_shape[1] - filter_shape[1] // self.stride) + 1
        self.outputSize = (self.layer_size,) + (x_ops,y_ops)
    def init(self,X):
        if X == 0:
            return
        x_ops = (X[1] - self.filter_shape[0] // self.stride) + 1 
        y_ops = (X[2] - self.filter_shape[1] // self.stride) + 1
        self.outputSize = (X[0] * self.layer_size,) + (x_ops,y_ops)
        pass
    def forward(self,_3Dvolumes): # Takes input in the form of 
        self.outputs = []
        for volume in _3Dvolumes:
            Convolutions = []
            for Slice in volume:
                for Filter in self.weights:
                    Filter = Filter.reshape(self.filter_shape)
                    Convolutions.append(self.activation(self.Convolve(Slice,Filter)))
            self.outputs.append(Convolutions)
        self.outputs = np.array(self.outputs)
        return self.outputs
    def backward(self,y): # Bad
        new_filters = []
        new_deltas = [[[] for _ in range(len(self.weights))] for i in range(len(self.next().delta))]
        for i in range(len(self.weights)):
            Filter = self.weights[i]
            new_deltas2 = [] 
            for j in range(len(self.next().delta)):
                volume = self.next().delta[j]
                convs = []
                Delta_Convolutions = []
                for delta in volume:
                    delta_in, convset = self.Convolve(delta,Filter.reshape(self.filter_shape),store=True)
                    Delta_Convolutions.append(delta_in)
                    convs += convset
                new_deltas[j][i] = np.average(Delta_Convolutions,0)
            #new_deltas.append(new_deltas2)
            new_filters.append(np.average(convs,0).flatten())
        self.weights = np.array(new_filters)
        self.delta = np.array(new_deltas)
        #print(self.next().delta.shape,self.delta.shape)
    def Convolve(self,img,_filter,store = False):
        k = _filter.shape
        offset = int(k[0]/2)
        x,y = offset,offset
        # NOTE: These op counts are unchecked 
        x_ops = (img.shape[0] - k[0] // self.stride) + 1 
        y_ops = (img.shape[1] - k[1] // self.stride) + 1
        final = np.zeros((x_ops,y_ops))
        convs = []
        for _y in range(y_ops):
            for _x in range(x_ops):
                conv = _filter * img[y-offset:y+offset+1][:,x-offset:x+offset+1]
                final[_y][_x] = sum(sum(conv))
                convs.append(conv)
                x+=self.stride
            x = offset
            y+=self.stride
        if store:
            return final, convs
        return final
    def description(self): # Provide String representation to store the model 
        pass
