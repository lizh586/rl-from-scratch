from engine import Tensor
import random

class Neuron:
    def __init__(self, nin, activation='relu'):
        # 随机初始化 nin 个 weight（Tensor），1 个 bias（Tensor）
        self.weight = [Tensor(random.uniform(-1,1)) for _ in range(nin)]
        self.bias = Tensor(random.uniform(-1,1))
        # 存 activation 选择（relu / linear）
        self.activation = activation

    def __call__(self, x):
        # x 是 list of Tensor（或 list of float，先转成 Tensor）
        out = Tensor(0)
        for i in range (len(x)):
            out += (self.weight[i] * x[i])
        out += self.bias
        if self.activation == 'relu':
            out = out.relu()
        return out
    
    def parameters(self):
        return self.weight + [self.bias]

class Layer:
    def __init__(self, nin, nout, activation='relu'):
        self.neuron = [Neuron(nin,activation) for _ in range(nout)]
    
    def __call__(self, x):
        out = []
        for neu in self.neuron:
            out.append(neu(x))
        return out
    
    def parameters(self):
        parameters = []
        for neu in self.neuron:
            parameters.extend(neu.parameters())
        return parameters
    
class MLP:
    def __init__(self, nin, nouts):
        self.layers = []
        pre = nin
        for i in range(len(nouts)):
            curr = nouts[i]
            if i != len(nouts) - 1:
                self.layers.append(Layer(pre, curr,'relu'))
            else:
                self.layers.append(Layer(pre, curr, 'linear'))
            pre = curr
    
    def __call__(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out
    
    def parameters(self):
        parameters = []
        for layer in self.layers:
            parameters.extend(layer.parameters())
        return parameters