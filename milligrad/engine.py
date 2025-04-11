from __future__ import annotations
import itertools
import math
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple, Union, Set, List, Iterator


ValueLike = Union[int, float, 'Value']


@dataclass
class ValueOrigin:
    '''
    ValueOrigin stores the operations used to create the current Value,
    as well as the parent Values from which it was derived

    Attributes:
        parents: tuple of parent Values that created the current Value
        backward_fn: function called during backpropagation for gradients
        op_name: name of the operation that created the current Value
    '''

    parents: Tuple[Value, ...] = ()
    backward_fn: Optional[Callable] = None
    op_name: Optional[str] = None


class Value:
    '''
    A scalar value that supports automatic differentiation.
    This class implements a computation graph that tracks operations performed
    on Values, enabling automatic computation of gradients through
    backpropagation. Each Value stores:
    - Its scalar data
    - Its gradient (computed during backward pass)
    - Its computation history

    Values can be combined using standard arithmetic
    operations (+, -, *, /, **) and other functions
    (exp, log, sigmoid, relu, tanh). Each operation creates a new Value
    and records how it was derived

    Example:
        x = Value(2.0)
        y = Value(-4.0)
        z = x * y + x.exp()  # Creates computation graph
        z.backward()  # Computes gradients for x and y
        print(x.grad)  # Shows gradient of z w.r.t. x
    '''

    _id_counter: Iterator[int] = itertools.count()

    def __init__(
        self,
        data: Union[int, float],
        name: Optional[str] = None,
        requires_grad: bool = True
    ) -> None:
        '''
        Initialize a Value object

        Args:
            data: scalar value to be stored
            name: optional name for the Value, defaults to its
                unique ID as string
            requires_grad: if True, gradients will be computed for this Value
        '''
        if not isinstance(data, (int, float)):
            msg = (f'Expected type float or int, got: '
                   f'{data} with type {type(data).__name__}')
            raise TypeError(msg)
        self.unique_id: int = next(self._id_counter)
        self.data: float = float(data)
        self.name: str = name if name is not None else str(self.unique_id)
        self.requires_grad: bool = requires_grad
        self.grad: float = 0.0
        self._origin: ValueOrigin = ValueOrigin()

    def __add__(self, other: ValueLike) -> Value:
        Value._validate_type(other, (int, float, Value), '+')
        other = Value._wrap_other(other)
        out = Value(self.data + other.data,
                    requires_grad=self.requires_grad or other.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += 1.0 * out.grad
            if other.requires_grad:
                other.grad += 1.0 * out.grad
        out._origin = ValueOrigin((self, other), _backward, '+')

        return out

    def __mul__(self, other: ValueLike) -> Value:
        Value._validate_type(other, (int, float, Value), '*')
        other = Value._wrap_other(other)
        out = Value(self.data * other.data,
                    requires_grad=self.requires_grad or other.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += other.data * out.grad
            if other.requires_grad:
                other.grad += self.data * out.grad
        out._origin = ValueOrigin((self, other), _backward, '*')

        return out

    def __pow__(self, other: Union[int, float]) -> Value:
        '''Supports only int/float powers'''
        Value._validate_type(other, (int, float), '**')
        out = Value(self.data**other,
                    requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += other * self.data**(other-1) * out.grad
        out._origin = ValueOrigin((self, ), _backward, f'**{other}')

        return out

    def __neg__(self) -> Value:
        return self * -1

    def __radd__(self, other: ValueLike) -> Value:
        return self + other

    def __rmul__(self, other: ValueLike) -> Value:
        return self * other

    def __sub__(self, other: ValueLike) -> Value:
        Value._validate_type(other, (int, float, Value), '-')
        return self + (-other)

    def __rsub__(self, other: ValueLike) -> Value:
        Value._validate_type(other, (int, float, Value), '-')
        return other + (-self)

    def __truediv__(self, other: ValueLike) -> Value:
        Value._validate_type(other, (int, float, Value), '/')
        if (isinstance(other, Value) and other.data == 0) or other == 0:
            raise ZeroDivisionError('Division by zero')
        return self * other**-1

    def __rtruediv__(self, other: ValueLike) -> Value:
        Value._validate_type(other, (int, float, Value), '/')
        if self.data == 0:
            raise ZeroDivisionError('Division by zero')
        return other * self**-1

    def __repr__(self) -> str:
        return f'Value({self.data}, name: {self.name})'

    def exp(self) -> Value:
        out = Value(math.exp(self.data),
                    requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += out.data * out.grad
        out._origin = ValueOrigin((self, ), _backward, 'exp')

        return out

    def log(self) -> Value:
        '''
        We add small epsilon to avoid log(0) and raise ValueError
        if it's still non-positive
        '''
        if self.data <= 0:
            eps = 1e-8
            self.data += eps
        if self.data <= 0:
            raise ValueError(f'Log of non-positive value {self.data}')
        out = Value(math.log(self.data),
                    requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += out.grad / self.data
        out._origin = ValueOrigin((self, ), _backward, 'log')

        return out

    def sigmoid(self) -> Value:
        '''More stable implementation of the sigmoid function'''
        if self.data >= 0:
            z = math.exp(-self.data)
            sigm = 1 / (1 + z)
        else:
            z = math.exp(self.data)
            sigm = z / (1 + z)
        out = Value(sigm,
                    requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += sigm * (1 - sigm) * out.grad
        out._origin = ValueOrigin((self, ), _backward, 'sigmoid')

        return out

    def relu(self) -> Value:
        '''ReLU (rectified linear unit) function'''
        out = Value(self.data if self.data > 0 else 0.0,
                    requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += (1.0 if out.data > 0 else 0.0) * out.grad
        out._origin = ValueOrigin((self, ), _backward, 'relu')

        return out

    def tanh(self) -> Value:
        '''Hyperbolic tangent function'''
        t = (math.exp(2*self.data) - 1)/(math.exp(2*self.data) + 1)
        out = Value(t,
                    requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += (1 - t**2) * out.grad
        out._origin = ValueOrigin((self, ), _backward, 'tanh')

        return out

    def backward(self, retain_graph: bool = False) -> None:
        '''
        Computes gradients for all leaves in the computation graph.
        Requires the output to be a scalar, such as a loss function.
        Gradients accumulate by default, so you may need to reset them
        manually, e.g. using Value.zero_grad() or optimizer.zero_grad()
        to reset gradients of all parent leaves as well

        Args:
            retain_graph: if False (set by default), clears the computation
                graph after backward pass to avoid memory leaks while
                preserving gradients. If True, keeps the graph for
                subsequent passes
        '''
        if not self.requires_grad:
            msg = 'Cannot call backward() on Value with requires_grad=False'
            raise RuntimeError(msg)

        topo: List[Value] = []
        visited: Set[int] = set()

        def build_topo(val: Value) -> None:
            '''
            Computes the topological order of the computation graph.
            Leaf nodes are not included since we don't
            backpropagate through them
            '''
            if val.unique_id in visited:
                return
            if val.is_leaf:
                return
            for parent in val.origin.parents:
                build_topo(parent)
            visited.add(val.unique_id)
            topo.append(val)
        build_topo(self)

        # Back propagation
        self.grad = 1.0
        for val in reversed(topo):
            if val.origin.backward_fn is not None:
                val.origin.backward_fn()

        # Remove all references to parent values to avoid memory leaks
        if not retain_graph:
            for val in topo:
                val._origin = ValueOrigin()

    def requires_grad_(self, req_grad: bool) -> None:
        '''Enable or disable gradient tracking for this Value'''
        if not self.is_leaf:
            msg = 'You can only change requires_grad flags of leaf Values'
            raise RuntimeError(msg)
        self.requires_grad = req_grad

    def zero_grad(self) -> None:
        '''Reset the gradient of the current Value to zero'''
        self.grad = 0.0

    @property
    def is_leaf(self) -> bool:
        '''
        Determines if the Value is a leaf node in the computation graph.
        A Value is a leaf if it either:
        1. Has requires_grad=False, or
        2. Has requires_grad=True and was created directly by the user
           (not the result of an operation, so backward_fn is None)
        '''
        return not self.requires_grad or self.origin.backward_fn is None

    @property
    def origin(self) -> ValueOrigin:
        '''
        Get the ValueOrigin object storing information about
        how this Value was created
        '''
        return self._origin

    @staticmethod
    def _validate_type(x: Any,
                       allowed_types: Union[type, Tuple[type, ...]],
                       operation_name: str = 'operation') -> None:
        '''
        Internal helper for type checking in operations.
        Raises TypeError if x is not an instance of any allowed_types

        Args:
            x: Value to check
            allowed_types: type or tuple of types that x should be
            operation_name: name of the operation for error messages
        '''
        if not isinstance(x, allowed_types):
            if isinstance(allowed_types, type):
                types_names = allowed_types.__name__
            else:
                types_names = ', '.join(t.__name__ for t in allowed_types)
            msg = (f'unsupported operand type(s) for {operation_name}: '
                   f'"Value" and "{type(x).__name__}". '
                   f'Expected type(s): {types_names}')
            raise TypeError(msg)

    @staticmethod
    def _wrap_other(other: ValueLike) -> Value:
        '''
        Convert constants to Value objects with requires_grad=False.
        E.g., "Value(4.0) + 2" where Value(4.0) is self and 2 is other,
        2 is considered a constant. If other is already a Value,
        returns it unchanged
        '''
        return (other if isinstance(other, Value)
                else Value(other, requires_grad=False))
