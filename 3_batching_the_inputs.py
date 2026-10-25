# so now we want to something smarter. Insted of going 1 by to the inputs, we want to batch them all at once.
# we are only going to add one dimension so it is easy to follow

import random

# here is all the unchanging stuff

def exp(value):
    return 2.718281828459045 ** value

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

alpha = 0.001 

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


# Now what needs to change

# added this due to how much of it is needed now
def copy_md_list(X):
    return [x for x in X]

class Activator:
    def __init__(self, func, derivative):
        self.func = func
        self.derivative = derivative
        self.last_input_batch = None

    # now takes multidim

    def forward(self, X):
        self.last_input_batch = copy_md_list(X)
        return [list(map(self.func, last_input)) for last_input in self.last_input_batch]
    
    def backward(self, upstream_gradient_batch, learning_rate):
            out = [
                [
                    grad * self.derivative(z)
                    for grad, z in zip(upstream_gradient, last_input)
                ]
                for last_input, upstream_gradient in zip(self.last_input_batch, upstream_gradient_batch)
            ]
            return out
    
    def copy(self):
        return Activator(self.func, self.derivative)

    def __call__(self, X):
        return self.forward(X)


sigmoid_activator = Activator(sigmoid, sigmoid_derivative)
tanh_activator = Activator(tanh, tanh_derivative)
relu_activator = Activator(relu, relu_derivative)
leaky_relu_activator = Activator(leaky_relu, leaky_relu_derivative)
elu_activator = Activator(elu, elu_derivative)

class Layer:
    def __init__(self, input_size, output_size):
        self.weight = [[random.gauss(0, 1)*0.1 for _ in range(input_size)] for _ in range(output_size)]
        self.bias = [0 for _ in range(output_size)]
        self.input_size = input_size
        self.output_size = output_size
        self.last_input_batch = None

    @staticmethod
    def dot_product(vector1, vector2) -> float:
        assert len(vector1) == len(vector2)
        return float(sum(x*y for x,y in zip(vector1, vector2)))
    
    def forward(self, X):
        # reimplemented as self contained for loop. Cool, huh?
        self.last_input_batch = copy_md_list(X)
        output = [
            [
                self.dot_product(neuron, last_input) + self.bias[i]
                for i, neuron in enumerate(self.weight)
            ] for last_input in self.last_input_batch
        ]
        return output
    
    # we compressed the backprop so now its just a simple as 1 and 2
    def backward(self, upstream_gradient_batch, learning_rate): 
        assert len(upstream_gradient_batch[0]) == self.output_size, "mismatch gradient and layer size"
        w_grad = [[0.0]*self.input_size for _ in range(self.output_size)]
        b_grad = [0.0]*self.output_size
        input_gradient_batch = []
        for last_input, upstream in zip(self.last_input_batch, upstream_gradient_batch):
            input_gradient = [0.0]*self.input_size
            for i, (neuron, g) in enumerate(zip(self.weight, upstream)):
                for j, w in enumerate(neuron):
                    input_gradient[j] += g * w          # uses the OLD weights
                    w_grad[i][j] += g * last_input[j]
                b_grad[i] += g
            input_gradient_batch.append(input_gradient)

        
        n = len(upstream_gradient_batch)
        for i in range(self.output_size):
            for j in range(self.input_size):
                self.weight[i][j] -= learning_rate * w_grad[i][j] / n
            self.bias[i] -= learning_rate * b_grad[i] / n
        return input_gradient_batch

    def __call__(self, X):
        return self.forward(X)


class Softmax:
    def forward(self, X):
        out_batch = []
        for x in X:
            Z = list(map(exp, x))
            s = sum(Z)
            out_batch.append([z/s for z in Z])
        return out_batch
    
    def __call__(self, X):
        return self.forward(X)

class Neural_Net:
    def __init__(self, input_size, *hidden_layers, output_size=None, activation_func=None):
        self.last_output_batch = None
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
        self.last_input_batch = copy_md_list(X)
        y = X
        for layer in self.net:
            y = layer(y)
        self.last_output_batch = self.output_func(y)
        return self.last_output_batch
    
    def backward(self, true_batch, learning_rate):
        grad = [
            [
                (out - true_per)
                for out, true_per in zip(last_output, true)
            ] for true, last_output, last_input in zip(true_batch, self.last_output_batch, self.last_input_batch)
        ] 
        for layer in self.net[::-1]:
            grad = layer.backward(grad, learning_rate)

    def __call__(self, X):
        return self.forward(X)

#added this for simplicity sake
def get_acc(network, X, Y):
    return [argmax(y_) == y for y_, y in zip(network(X), Y)]
        
# in the end we sacrifice some model performance but we do get
# benefits in actual perf
if __name__ == "__main__":
    random.seed(42)
    from sklearn.datasets import load_digits
    from time import perf_counter
    X,_y = load_digits(return_X_y=True); X = (X / 16).tolist()
    Y = [eye(obj, 10) for obj in _y]
    network = Neural_Net(64, 32, 10)
    start = perf_counter()
    results = get_acc(network, X, _y)
    print(f"prelearning {sum(results)/len(results)}")
    for epoch in range(300):
        #print(f"epoch: {epoch}")
        network(X)
        network.backward(Y, 0.5)
    end = perf_counter()
    results = get_acc(network, X, _y)
    print(f"postlearning {sum(results)/len(results)}; time {end - start}")