import numpy as np

## by yourself .Finish your own NN framework
## Just an example.You can alter sample code anywhere. 

class Conv2D(object):
    def __init__(self, input_size, n_filters, k_size):
        # input size
        self.input_size = input_size
        
        # number of filters
        self.n_filters = n_filters
        
        # kernel size
        self.k_size = k_size
        
        # filters is a 3d array w/ dimensions (n_filters, 3, 3)
        self.filters = np.random.randn(n_filters, k_size, k_size)
        
    
    def iterate_regions(self, img):
        img = img.reshape(self.input_size, self.input_size)
        h, w = img.shape
        
        for i in range(h-2):
            for j in range(w-2):
                img_region = img[i:(i+3), j:(j+3)]
                yield img_region, i, j
    
    def forward(self, input):
        input = input.reshape(self.input_size, self.input_size)
        self.last_input = input
        h, w = input.shape
        
        output = np.zeros((h-2, w-2, self.n_filters))
        
        # Convolution
        for img_region, i, j in self.iterate_regions(input):
            output[i,j] = np.sum(img_region * self.filters, axis=(1,2))
        
        return output

    def backward(self, output_grad, lr):
        dL_dfilters = np.zeros(self.filters.shape)
        
        for img_region, i, j in self.iterate_regions(self.last_input):
            for f in range(self.n_filters):
                dL_dfilters[f] += output_grad[i, j, f] * img_region
        
        self.filters -= lr * dL_dfilters
        
        return dL_dfilters[f]
    
class MaxPool2D(object):
    def iterate_region(self, img):
        h, w, _ = img.shape
        new_h = h // 2
        new_w = w // 2
        
        for i in range(new_h):
            for j in range(new_w):
                img_region = img[(i*2):(i*2+2), (j*2):(j*2+2)]
                yield img_region, i, j
    
    def forward(self, input):
        self.last_input = input
        
        h, w, n_filters = input.shape
        output = np.zeros((h //2, w//2, n_filters))
        
        for img_region, i, j in self.iterate_region(input):
            output[i,j] = np.amax(img_region, axis=(0,1))
        
        return output
    
    def backward(self, output_grad):
        dL_dinput = np.zeros(self.last_input.shape)
        
        for img_region, i, j in self.iterate_region(self.last_input):
            h, w, f = img_region.shape
            amax = np.amax(img_region, axis=(0,1))
            
            for i2 in range(h):
                for j2 in range(w):
                    for f2 in range(f):
                        if img_region[i2,j2,f2] == amax[f2]:
                            dL_dinput[i*2+i2, j*2+j2, f2] = output_grad[i, j, f2]
        
        return dL_dinput

class FullyConnected(object):
    # A standar fully-connected layer with sorftmax activation
    def __init__(self, input_len, nodes):
        # Divide by input_len to reduce the variance of initial value
        self.weights = np.random.rand(input_len, nodes) / input_len
        self.bias = np.zeros(nodes)
        
    def forward(self, input):
        self.last_input_shape = input.shape
        
        input = input.flatten()
        self.last_input = input
        
        input_len, nodes = self.weights.shape
        
        totals = np.dot(input, self.weights) + self.bias
        
        self.last_total = totals
        
        # Softmax 
        exp = np.exp(totals)
        
        return exp / np.sum(exp, axis=0)
    
    def backward(self, output_grad, lr):
        for i, grad in enumerate(output_grad):
            if grad == 0:
                continue
            
            # e^totals
            t_exp = np.exp(self.last_total)
            
            # Sum of all e^totals
            S = np.sum(t_exp)
            
            # Gradient of out[i] against totals
            epsilon = 1e-10
            dout_dt = -t_exp[i] * t_exp / (S**2 + epsilon)
            dout_dt[i] = t_exp[i] * (S - t_exp[i]) / (S**2 + epsilon)
            
            # Gradients of totals against weights/biases/input
            dt_dw = self.last_input
            dt_db = 1
            dt_dinputs = self.weights
            
            # Gradients of loss against totals
            dL_dt = grad * dout_dt
            
            # Gradients of loss against weights/ biases / input 
            dL_dw = dt_dw[np.newaxis].T @ dL_dt[np.newaxis]
            dL_db = dL_dt * dt_db
            dL_dinputs = dt_dinputs @ dL_dt
            
            # Update weights / biases
            self.weights -= lr * dL_dw
            self.bias -= lr * dL_db
            
            
            return dL_dinputs.reshape(self.last_input_shape)
    