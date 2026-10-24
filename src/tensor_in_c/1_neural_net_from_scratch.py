# x is a input of n values; x : [x1, x2, x3, ..., xn]
# w is a set of weights that should be the same size as x; w : [w1, w2, w3, ..., wn]
# b is a single value;
# math of a neuron : w dot x + b.

import random

def exp(value):
    return 2.718281828459045 ** value

class Neuron:
    def __init__(self, input_size):
        self.weights = [random.gauss(0, 1)*0.1 for _ in range(input_size)]
        self.input_size = input_size
        self.bias = 0
        # these 2 are needed for backprop
        self.last_input = None

    @staticmethod
    def dot_product(vector1, vector2) -> float:
        assert len(vector1) == len(vector2)
        return float(sum(x*y for x,y in zip(vector1, vector2)))

    def forward(self, X):
        self.last_input = list(X)
        return self.dot_product(self.weights, self.last_input) + self.bias

    def backward(self, upstream_gradient, learning_rate):
        assert self.last_input is not None, "forward must be called before backward"
        input_gradient = [upstream_gradient * weight for weight in self.weights]
        for i, value in enumerate(self.last_input):
            self.weights[i] -= learning_rate * upstream_gradient * value
        self.bias -= learning_rate * upstream_gradient
        return input_gradient

    #call just lets us call the forward when using the instance like a class 
    #(pytorch does something similar)
    def __call__(self, X):
        return self.forward(X)

# a layer a stack of neurons
# each neuron is processed separately and returns the neurons output as a vector the size of the number of neurons

class Layer:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.neurons = [
            Neuron(input_size)
            for _ in range(output_size)
        ]

    def append(self, neuron:Neuron) -> None:
        assert isinstance(neuron, Neuron), "neuron is not a Neuron"
        assert self.input_size is None or self.input_size == neuron.input_size
        self.neurons.append(neuron)
        self.output_size += 1

    def forward(self, X):
        X = list(X)
        out = [
            neuron(X)
            for neuron in self.neurons
        ]
        return out

    def backward(self, upstream_gradient, learning_rate):
        assert len(upstream_gradient) == self.output_size, "mismatch gradient and layer size"
        input_gradient = [0.0] * self.input_size
        for neuron, gradient in zip(self.neurons, upstream_gradient):
            neuron_input_gradient = neuron.backward(gradient, learning_rate)
            for i, value in enumerate(neuron_input_gradient):
                input_gradient[i] += value
        return input_gradient
    
    def __call__(self, X):
        return self.forward(X)

alpha = 0.001 

# non-linear functions used for activation.
# why are they necessary? Many problems are non-linear and not having these functions between means any 
# neural net can be decayed into a single linear layer
# 
# Example: XOR (not linearly separable)

def sigmoid(x):
    return 1/(1 + exp(-x))



def tanh(x):
    return (2 / (1 + exp(-2*x))) - 1

def relu(x):
    return max(0, x)

def leaky_relu(x):
    if x > 0:
        return x
    else:
        return alpha * x
    
def elu(x):
    if x > 0:
        return x
    else:
        return alpha * ( exp(x) - 1 )

# derivatives of the activator functions

def sigmoid_derivative(x):
    y = sigmoid(x)
    return y * (1 - y)

def tanh_derivative(x):
    y = tanh(x)
    return 1 - y * y

def relu_derivative(x):
    return 1 if x > 0 else 0

def leaky_relu_derivative(x):
    return 1 if x > 0 else alpha

def elu_derivative(x):
    return 1 if x > 0 else alpha * exp(x)


class Activator:
    def __init__(self, func, derivative):
        self.func = func
        self.derivative = derivative

    def forward(self, X):
        self.last_input = list(X)
        return list(map(self.func, self.last_input))
    
    def backward(self, upstream_gradient, learning_rate):
            out = [
                grad * self.derivative(z)
                for grad, z in zip(upstream_gradient, self.last_input)
            ]
            return out
    
    def copy(self):
        return Activator(self.func, self.derivative)

    def __call__(self, X):
        return self.forward(X)

# here we construct all the full activator function to use later for backprop
sigmoid_activator = Activator(sigmoid, sigmoid_derivative)
tanh_activator = Activator(tanh, tanh_derivative)
relu_activator = Activator(relu, relu_derivative)
leaky_relu_activator = Activator(leaky_relu, leaky_relu_derivative)
elu_activator = Activator(elu, elu_derivative)

# for prob we will use a simple soft max. A soft max take the exponential value of all the values of output as turn them
# into related probabilities

class Softmax:
    def forward(self, X):
        Z = list(map(exp, X))
        s = sum(Z)
        return [z/s for z in Z]
    
    def __call__(self, X):
        return self.forward(X)


# now we make the neural net (simplified)

class Neural_Net:
    def __init__(self, input_size, *hidden_layers, output_size=None, activation_func=None):
        self.last_input = None
        self.net = []
        self.backprop_calculator = []
        self.output_func = Softmax()
        if activation_func is None:
            activation_func = relu_activator
        sizes = [input_size] + list(hidden_layers) + [] if output_size is None else [output_size]
        for i, (inp, oup) in enumerate(zip(sizes, sizes[1:])):
            self.net.append(Layer(inp, oup))
            if i == len(sizes) - 2:
                pass
            else:
                self.net.append(activation_func.copy())
    
    def forward(self, X):
        self.last_input = list(X)
        y = X
        for layer in self.net:
            y = layer(y)
        self.last_output = self.output_func(y)
        return self.last_output
    
    def backward(self, true, learning_rate):
        grad = [
            (out - true_per) / len(self.last_input)
            for out, true_per in zip(self.last_output, true)
        ] 
        for layer in self.net[::-1]:
            grad = layer.backward(grad, learning_rate)

    def __call__(self, X):
        return self.forward(X)

def eye(index, shape):
    return [1 if i == index else 0 for i in range(shape)]

def argmax(l1):
    m = l1[0]
    i = 0
    for j, x in enumerate(l1):
        if x > m:
            i = j
            m = x
    return i
        
# since this isnt a actual implementation we can get away with testing on training data
# however in actual implementation you need to have a split between training and testing data to 
# show that you model has actually generalized and doesn't just work optimally for test data
if __name__ == "__main__":
    random.seed(42)
    from sklearn.datasets import load_digits
    from time import perf_counter
    X,_y = load_digits(return_X_y=True); X = X / 16
    Y = [eye(obj, 10) for obj in _y]
    network = Neural_Net(64, 32, 10)
    start = perf_counter()
    results = [argmax(network(x)) == y for x,y in zip(X,_y)]
    print(f"prelearning {sum(results)/len(results)}")
    for epoch in range(300):
        #print(f"epoch {epoch}")
        for x,y in zip(X,Y):
            network(x)
            network.backward(y, 0.5)
    end = perf_counter()
    results = [argmax(network(x)) == y for x,y in zip(X,_y)]
    print(f"postlearning {sum(results)/len(results)}; time {end - start}")
