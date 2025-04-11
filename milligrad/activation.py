from typing import List

from .engine import Value
from .nn import Module


class ReLU(Module):
    '''Applies the rectified linear unit function element-wise'''
    def forward(self, inputs: List[Value]) -> List[Value]:
        return [value.relu() for value in inputs]

    def __repr__(self) -> str:
        return 'ReLU()'


class Sigmoid(Module):
    '''Applies the sigmoid function element-wise'''
    def forward(self, inputs: List[Value]) -> List[Value]:
        return [value.sigmoid() for value in inputs]

    def __repr__(self) -> str:
        return 'Sigmoid()'


class Tanh(Module):
    '''Applies the hyperbolic tangent function element-wise'''
    def forward(self, inputs: List[Value]) -> List[Value]:
        return [value.tanh() for value in inputs]

    def __repr__(self) -> str:
        return 'Tanh()'


class Softmax(Module):
    '''
    Applies the Softmax function to a list of Values. Rescales them so that
    all Values of the output list lie in the range [0, 1] and sum to 1
    '''
    def forward(self, inputs: List[Value]) -> List[Value]:
        denominator = sum((val.exp() for val in inputs))
        return [val.exp() / denominator for val in inputs]

    def __repr__(self) -> str:
        return 'Softmax()'
