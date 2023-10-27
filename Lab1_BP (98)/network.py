from .layer import *

class Network(object):
    def __init__(self):
        ## by yourself .Finish your own NN framework
        ## Just an example.You can alter sample code anywhere. 
        self.conv1 = Conv2D(input_size=28, n_filters=20, k_size=3)
        self.pool1 = MaxPool2D()
        self.fc1 = FullyConnected(13*13*20, 10)
        self.loss = 0


    def forward(self, input, label):
        ## by yourself .Finish your own NN framework
        out = self.conv1.forward(input / 255.)
        out = self.pool1.forward(out)
        out = self.fc1.forward(out)
        
        epsilon = 10e-15
        loss = -np.sum(label * np.log(out + epsilon))
        
        return out, loss

    def backward(self, input, label, lr):
        ## by yourself .Finish your own NN framework
        # Forward
        out, loss = self.forward(input, label)
        
        # Calculate initial gradient
        grad = np.zeros(10)
        grad[np.argmax(label)] = -1 / out[np.argmax(label)]
        
        # Backpropagation
        grad = self.fc1.backward(grad, lr)
        grad = self.pool1.backward(grad)
        grad = self.conv1.backward(grad, lr)
        
        return loss