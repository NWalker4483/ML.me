Run any of the sample files "ae.py, mlp.py, cnn.py"
``` python 
X = flatten_img_list(X)
net = NeuralNetwork()
net.add(Dense(784))
net.add(Dense(16))
net.add(Dense(16))
net.add(Dense(10))

net.set_training_set(X[:3000],Y[:3000])
net.train(epochs = 10000,batch_size = 200)
net.save("MLP.model")
print(net.get_recall())
```