import math

class Tensor:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward 
        return out    
    
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self,other),'+')
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out    
    
    def __neg__(self):
        return self * (-1)

    def __sub__(self, other):
        return self + (-other)
    
    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data/other.data,(self, other), '/')
        def _backward():
            self.grad += 1/other.data * out.grad
            other.grad += - self.data / (other.data ** 2) * out.grad
        out._backward = _backward
        return out
        
    def exp(self):
        out = Tensor(math.exp(self.data),(self,),'exp')
        def _backward():
            self.grad += out.grad * out.data 
        out._backward = _backward
        return out

    def relu(self):
        data = self.data if self.data > 0.0 else 0.0

        out = Tensor(data,(self,),'relu')
        def _backward():
            grad = 1.0 if data > 0.0 else 0.0
            self.grad += grad * out.grad
        out._backward = _backward
        return out


    def backward(self):
        self.grad = 1.0
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        for v in reversed(topo):
            v._backward()

