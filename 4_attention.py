# so now we want to something smarter. Insted of going 1 by to the inputs, we want to batch them all at once.
# we are only going to add one dimension so it is easy to follow

import random
from generics_for_4_fowards import *

def dot_product(vector1, vector2) -> float:
    assert len(vector1) == len(vector2)
    return float(sum(x*y for x,y in zip(vector1, vector2)))

def linear(input_, weight, bias = None):
    if bias is None:
        return [
            [
                dot_product(row, vector)
                for i, row in enumerate(weight)
            ] for vector in input_
        ]
    else:
        return [
            [
                dot_product(row, vector) + bias[i]
                for i, row in enumerate(weight)
            ] for vector in input_
        ]

def attention_linear(key, query):
    dk = sqrt(len(key[0]))
    return [
        [
            dot_product(q, k) / dk if j <= i else float('-inf')
            for j, k in enumerate(key)
        ]
        for i, q in enumerate(query)
    ]


class Activator:
    def __init__(self, func, derivative):
        self.func = func
        self.derivative = derivative
        self.last_input_batch = None

    def forward(self, X):
        self.last_input_batch = copy_md_list(X)
        return [list(map(self.func, last_input)) for last_input in X]
    
    def backward(self, upstream_gradient_batch, learning_rate):
        return [
            [
                grad * self.derivative(z)
                for grad, z in zip(upstream_gradient, last_input)
            ]
            for last_input, upstream_gradient in zip(self.last_input_batch, upstream_gradient_batch)
        ]
    
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
    
    def forward(self, X):
        # reimplemented as self contained for loop. Cool, huh?
        self.last_input_batch = copy_md_list(X)
        return linear(X, self.weight, self.bias)
    
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


def softmax(X):
    out_batch = []
    for x in X:
        Z = list(map(exp, x))
        s = sum(Z)
        out_batch.append([z/s for z in Z])
    return out_batch

def scaled_dot_product_attention(K, Q, V):
    attention = softmax(attention_linear(K, Q))
    return linear(attention, V)

class Head:
    def __init__(self, input_size, attention_size):
        self.dk = attention_size
        self.W_q = Layer(input_size, attention_size)
        self.W_k = Layer(input_size, attention_size)
        self.W_v = Layer(input_size, attention_size)
        self.last_input_batch = None

    def forward(self, X):
        # here we are going to assume query, key and value are all the input

        self.last_input_batch = copy_md_list(X)
        Q = self.W_q(X)
        K = self.W_k(X)
        V = self.W_v(X)

        attention_weights = softmax(attention_linear(K, Q))
        output = linear(attention_weights, V)

        self.last_Q = copy_md_list(Q)
        self.last_K = copy_md_list(K)
        self.last_A = copy_md_list(attention_weights)
        self.last_V = copy_md_list(V)
        
        return output

    def backward(self, X):
        pass

    def __call__(self, X):
        return self.forward(X)
        


class MultiHead:
    def __init__(self, input_size, num_heads=4, attention_size=None, output_size=None):
        if attention_size is None:
            attention_size = input_size
        if output_size is None:
            output_size = input_size
        self.num_heads = num_heads
        self.heads = [Head(input_size, attention_size) for _ in range(4)]
        self.W_o = Layer(self.num_heads*attention_size, output_size)
    
    def concat(*batches):
        out = []
        for batch in batches:
            for item in batch:
                out.append(item)
        return out

    def forward(self, X):
        pre_out = concat(
            *[head(X) for head in self.heads]
        )
        return self.W_o(pre_out)
    
    def backward(self, upstream_gradient_batch, learning_rate):
        d_concat = self.W_o.backward(upstream_gradient_batch, learning_rate)

        d_heads = []

        #resplit
        for h in range(self.num_heads):
            start = h * self.attention_size
            end = start + self.attention_size
            d_heads.append([
                row[start:end]
                for row in d_concat
            ])

        dX = [
            [0.0] * self.heads[0].W_q.input_size
            for _ in range(len(d_concat))
        ]
        
        for head, d_head in zip(self.heads, d_heads):
            head_dX = head.backward(d_head, learning_rate)
            for t, row in enumerate(head_dX):
                for j, value in enumerate(row):
                    dX[t][j] += value

        return dX

    
    def __call__(self, X):
        return self.forward(X)


        



#added this for simplicity sake
def get_acc(network, X, Y):
    return [argmax(y_) == y for y_, y in zip(network(X), Y)]
        
# in the end we sacrifice some model performance but we do get
# benefits in actual perf
if __name__ == "__main__":
    pass
