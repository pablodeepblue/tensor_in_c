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
        if activation_func is None:
            activation_func = relu_activator
        sizes = [input_size] + list(hidden_layers) + ([output_size] if output_size is not None else [])
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
        return y
    
    # make nn a generic
    def backward(self, grad, learning_rate):
        for layer in self.net[::-1]:
            grad = layer.backward(grad, learning_rate)
        return grad

    def __call__(self, X):
        return self.forward(X)


def softmax(X):
    out_batch = []
    for x in X:
        Z = list(map(exp, x))
        s = sum(Z)
        out_batch.append([z/s for z in Z])
    return out_batch

def transpose(M):
    return [list(r) for r in zip(*M)]

def sum_2d(*matrices):
    return [[sum(vals) for vals in zip(*rows)] for rows in zip(*matrices)]

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

        A = softmax(attention_linear(K, Q))
        output = linear(A, transpose(V))

        self.last_Q = copy_md_list(Q)
        self.last_K = copy_md_list(K)
        self.last_A = copy_md_list(A)
        self.last_V = copy_md_list(V)
        
        return output

    def backward(self, upstream_gradient_batch, learning_rate):
        Q = self.last_Q
        K = self.last_K
        V = self.last_V
        A = self.last_A

        dO = upstream_gradient_batch

        batch_len = len(A)
        seq_len = batch_len
        dk = self.dk
        scale = 1.0 / sqrt(dk)

        dA = [
            [
                sum(dO[t][j] * V[s][j] for j in range(dk))
                for s in range(batch_len)
            ]
            for t in range(batch_len)
        ]

        dV = [
            [
                sum(A[t][s] * dO[t][j] for t in range(batch_len))
                for j in range(dk)
            ]
            for s in range(batch_len)
        ]

        dS = []

        for t in range(seq_len):
            row_dot = sum(
                A[t][s] * dA[t][s]
                for s in range(seq_len)
            )

            dS.append([
                A[t][s] * (dA[t][s] - row_dot)
                for s in range(seq_len)
            ])

        dQ = [
            [
                scale * sum(
                    dS[t][s] * K[s][j]
                    for s in range(seq_len)
                )
                for j in range(dk)
            ]
            for t in range(seq_len)
        ]

        dK = [
            [
                scale * sum(
                    dS[t][s] * Q[t][j]
                    for t in range(seq_len)
                )
                for j in range(dk)
            ]
            for s in range(seq_len)
        ]

        dX_q = self.W_q.backward(dQ, learning_rate)
        dX_k = self.W_k.backward(dK, learning_rate)
        dX_v = self.W_v.backward(dV, learning_rate)


        dX = [
            [
                dX_q[t][j] + dX_k[t][j] + dX_v[t][j]
                for j in range(len(dX_q[t]))
            ]
            for t in range(len(dX_q))
        ]

        return dX

    def __call__(self, X):
        return self.forward(X)
        


class MultiHead:
    def __init__(self, input_size, num_heads=4, attention_size=None, output_size=None):
        if attention_size is None:
            attention_size = input_size
        if output_size is None:
            output_size = input_size
        self.attention_size = attention_size
        self.num_heads = num_heads
        self.heads = [Head(input_size, attention_size) for _ in range(num_heads)]
        self.W_o = Layer(self.num_heads*attention_size, output_size)
    
    @staticmethod
    def concat(*batches):
        return [sum(rows, []) for rows in zip(*batches)]

    def forward(self, X):
        pre_out = self.concat(
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


class LayerNorm:
    def __init__(self, size, epsilon=1e-5):
        self.size = size
        self.epsilon = epsilon
        self.gamma = [1.0] * size
        self.beta = [0.0] * size
        self.last_normalized = None
        self.last_inverse_std = None

    def forward(self, X):
        self.last_normalized = []
        self.last_inverse_std = []
        output = []
        for row in X:
            mean = sum(row) / self.size
            variance = sum((value - mean) ** 2 for value in row) / self.size
            inverse_std = 1.0 / sqrt(variance + self.epsilon)
            normalized = [(value - mean) * inverse_std for value in row]
            self.last_normalized.append(normalized)
            self.last_inverse_std.append(inverse_std)
            output.append([self.gamma[j] * value + self.beta[j] for j, value in enumerate(normalized)])
        return output

    def backward(self, upstream_gradient_batch, learning_rate):
        gamma_gradient = [0.0] * self.size
        beta_gradient = [0.0] * self.size
        input_gradient = []
        for normalized, inverse_std, upstream in zip(self.last_normalized, self.last_inverse_std, upstream_gradient_batch):
            scaled_gradient = [upstream[j] * self.gamma[j] for j in range(self.size)]
            mean_gradient = sum(scaled_gradient) / self.size
            normalized_gradient = sum(scaled_gradient[j] * normalized[j] for j in range(self.size)) / self.size
            input_gradient.append([
                inverse_std * (scaled_gradient[j] - mean_gradient - normalized[j] * normalized_gradient)
                for j in range(self.size)
            ])
            for j in range(self.size):
                gamma_gradient[j] += upstream[j] * normalized[j]
                beta_gradient[j] += upstream[j]
        count = len(upstream_gradient_batch)
        for j in range(self.size):
            self.gamma[j] -= learning_rate * gamma_gradient[j] / count
            self.beta[j] -= learning_rate * beta_gradient[j] / count
        return input_gradient

    def __call__(self, X):
        return self.forward(X)


class TransformerBlock:
    def __init__(self, att_input_size, att_output_size, mlp_output_size, num_heads=4, feed_forward_size=None):
        if feed_forward_size is None:
            feed_forward_size = 4 * att_output_size
        self.attention = MultiHead(att_input_size, num_heads=num_heads, output_size=att_output_size)
        self.attention_residual = Layer(att_input_size, att_output_size) if att_input_size != att_output_size else None
        self.attention_norm = LayerNorm(att_output_size)
        self.mlp = Neural_Net(att_output_size, feed_forward_size, mlp_output_size)
        self.mlp_residual = Layer(att_output_size, mlp_output_size) if att_output_size != mlp_output_size else None
        self.mlp_norm = LayerNorm(mlp_output_size)

    def forward(self, X):
        residual = X if self.attention_residual is None else self.attention_residual(X)
        attention_output = self.attention(X)
        attention_output = [[a + b for a, b in zip(row_a, row_b)] for row_a, row_b in zip(attention_output, residual)]
        normalized_attention = self.attention_norm(attention_output)
        mlp_residual = normalized_attention if self.mlp_residual is None else self.mlp_residual(normalized_attention)
        mlp_output = self.mlp(normalized_attention)
        return self.mlp_norm([[a + b for a, b in zip(row_a, row_b)] for row_a, row_b in zip(mlp_output, mlp_residual)])

    def backward(self, upstream_gradient_batch, learning_rate):
        grad = self.mlp_norm.backward(upstream_gradient_batch, learning_rate)
        grad_residual = copy_md_list(grad)
        grad_mlp = self.mlp.backward(grad, learning_rate)
        grad_mlp_residual = grad_residual if self.mlp_residual is None else self.mlp_residual.backward(grad_residual, learning_rate)
        grad_attention_input = [
            [a + b for a, b in zip(row_a, row_b)]
            for row_a, row_b in zip(grad_mlp_residual, grad_mlp)
        ]
        grad_attention_input = self.attention_norm.backward(grad_attention_input, learning_rate)
        grad_residual = copy_md_list(grad_attention_input)
        grad_attention = self.attention.backward(grad_attention_input, learning_rate)
        if self.attention_residual is None:
            grad_residual_input = grad_residual
        else:
            grad_residual_input = self.attention_residual.backward(grad_residual, learning_rate)
        return [
            [a + b for a, b in zip(row_a, row_b)]
            for row_a, row_b in zip(grad_attention, grad_residual_input)
        ]

    def __call__(self, X):
        return self.forward(X)


#generated tests
if __name__ == "__main__":
    random.seed(42)

    width = 4
    classes = 2
    sequence_length = 6
    steps = 10000
    learning_rate = 0.03
    block = TransformerBlock(width, width, classes, num_heads=2)

    def make_example():
        values = [[random.gauss(0, 1) for _ in range(width)] for _ in range(sequence_length)]
        targets = [[1, 0] if row[0] < 0 else [0, 1] for row in values]
        return values, targets

    def evaluate(sample_count=100):
        correct = 0
        count = 0
        for _ in range(sample_count):
            values, targets = make_example()
            output = block(values)
            correct += sum(max(range(classes), key=row.__getitem__) == max(range(classes), key=target.__getitem__) for row, target in zip(output, targets))
            count += len(targets)
        return correct / count

    from time import perf_counter

    before = evaluate()
    start = perf_counter()
    for _ in range(steps):
        values, targets = make_example()
        output = block(values)
        gradient = [[a - b for a, b in zip(out_row, target_row)] for out_row, target_row in zip(output, targets)]
        block.backward(gradient, learning_rate)
    after = evaluate()
    elapsed = perf_counter() - start

    print(f"prelearning {before}")
    print(f"postlearning {after}; time {elapsed}")
