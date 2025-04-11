from __future__ import annotations
import math
import random
import warnings
from typing import Union, Any, List, Dict, Iterator

from .engine import Value, ValueLike


class Parameter:
    '''
    Container for trainable parameters in neural network modules.
    Wraps a list of Values and provides validation and access methods.
    '''

    def __init__(self, params: List[Value]) -> None:
        self._validate(params)
        self._params = params

    def _validate(self, params: List[Value]) -> None:
        '''Validate that params is a list of Values'''
        if not isinstance(params, list):
            msg = (f'Parameter must be list of Values, '
                   f'got {type(params).__name__} instead')
            raise TypeError(msg)
        if all(isinstance(val, Value) for val in params):
            return
        raise TypeError('Parameter must be list of Values')

    def __iter__(self) -> Iterator[Value]:
        yield from self._params

    def __getitem__(self, idx):
        return self._params[idx]

    def __repr__(self) -> str:
        return f'Parameter(shape: {self.shape})'

    @property
    def params(self) -> List[Value]:
        '''Get list of parameter Values'''
        return self._params

    @params.setter
    def params(self, new_params: List[Value]) -> None:
        '''Set new parameter Values with validation'''
        self._validate(new_params)
        if len(new_params) != self.shape:
            msg = (f'Parameter shape changed from {self.shape} to '
                   f'{len(new_params)}')
            warnings.warn(msg)
        self._params = new_params

    @property
    def shape(self) -> int:
        '''Get number of parameters'''
        return len(self.params)


class Module:
    '''
    Base class for all neural network modules.
    Every custom module should inherit from this class and implement:
    1. __init__: initialize parameters and submodules
    2. forward: define the computation performed at every call

    Example:
        class MyModule(Module):
            def __init__(self):
                super().__init__()
                self.linear = Linear(20, 4)
                self.activation = ReLU()

            def forward(self, x):
                x = self.linear(x)
                x = self.activation(x)
                return x
    '''

    def __init__(self) -> None:
        self._modules: Dict[str, Module] = {}
        self._parameters: Dict[str, Parameter] = {}

    def __call__(self, *inputs: Any) -> Any:
        return self.forward(*inputs)

    def __setattr__(
        self,
        name: str,
        value: Union[Parameter, Module, Any]
    ) -> None:
        if isinstance(value, Parameter):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        super().__setattr__(name, value)

    def forward(self, *inputs: Any) -> Any:
        '''
        Defines computation performed at every call.
        Should be overridden by all subclasses.
        '''
        raise NotImplementedError('Subclasses must implement forward()')

    def modules(self) -> List[Module]:
        '''Returns list of child modules'''
        return list(self._modules.values())

    def parameters(self) -> List[Value]:
        '''Returns list of all parameters in the module and submodules'''
        params = []

        def get_parameters(module: Module) -> None:
            for p in module._parameters.values():
                params.extend(p.params)
            for m in module._modules.values():
                get_parameters(m)

        get_parameters(self)
        return params


class Linear(Module):
    '''
    Applies a linear transformation to the input

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: if True, adds a learnable bias to the output
    '''

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True
    ) -> None:
        super().__init__()
        if in_features <= 0:
            msg = f'Size of input should be positive: {in_features}'
            raise ValueError(msg)
        if out_features <= 0:
            msg = f'Size of output should be positive: {out_features}'
            raise ValueError(msg)
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.reset_parameters()

    def reset_parameters(self) -> None:
        '''Initialize weights and biases'''

        bound = math.sqrt(1 / self.in_features)
        weights = [Value(random.uniform(-bound, bound))
                   for _ in range(self.in_features * self.out_features)]
        self.weights = Parameter(weights)

        if self.bias:
            biases = [Value(random.uniform(-bound, bound))
                      for _ in range(self.out_features)]
            self.biases = Parameter(biases)
        else:
            self.biases = None

    def forward(self, inputs: List[ValueLike]) -> List[Value]:
        if len(inputs) != self.in_features:
            msg = (f'Input size mismatch. Expected {self.in_features} '
                   f'features, got {len(inputs)}')
            raise ValueError(msg)

        outputs = []
        for i in range(self.out_features):
            # Get the weights for single neuron
            start_idx = i * self.in_features
            end_idx = start_idx + self.in_features
            neuron_weights = self.weights[start_idx:end_idx]
            weighted_sum = sum(w * x for w, x in zip(neuron_weights, inputs))
            outputs.append(weighted_sum)

        if self.biases is not None:
            outputs = [w_sum + b for w_sum, b in zip(outputs, self.biases)]

        return outputs

    def __repr__(self) -> str:
        return (f'Linear({self.in_features}, {self.out_features}, '
                f'bias={self.bias})')


class Sequential(Module):
    '''
    A sequential container for chaining modules in a sequence.
    Modules will be added to it in the order they
    are passed in the constructor

    Example:
        model = Sequential(
            Linear(20, 10),
            ReLU(),
            Linear(10, 1),
            Sigmoid()
        )
    '''

    def __init__(self, *layers: Module) -> None:
        super().__init__()
        for i, layer in enumerate(layers):
            if not isinstance(layer, Module):
                raise TypeError(f'Layer {layer} is not a Module')
            layer_name = f'{type(layer).__name__.lower()}_{i}'
            self._modules[layer_name] = layer
            # Also set as attribute for direct access
            setattr(self, layer_name, layer)

    def forward(self, x: Any) -> Any:
        for layer in self._modules.values():
            x = layer(x)
        return x

    def __repr__(self) -> str:
        module_repr = ',\n  '.join(str(m) for m in self.modules())
        return f'Sequential(\n  {module_repr}\n)'
