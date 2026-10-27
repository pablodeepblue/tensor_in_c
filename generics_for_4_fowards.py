alpha = 0.001

def exp(value):
    return 2.718281828459045 ** value

def sqrt(value):
    return value ** 0.5

def eye(index, shape):
    return [1 if i == index else 0 for i in range(shape)]

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

def copy_md_list(X):
    return [copy_md_list(x) if isinstance(x, list) else x for x in X]

def argmax(l1):
    m = l1[0]
    i = 0
    for j, x in enumerate(l1):
        if x > m:
            i = j
            m = x
    return i