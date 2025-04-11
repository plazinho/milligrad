from typing import List, Dict

from .engine import Value


class Optimizer:
    '''
    Base class for all optimizers.
    Optimizers update parameter values by subtracting scaled gradients
    to minimize a loss function

    Args:
        params: list of Values to optimize using their gradients.
                Only parameters with requires_grad=True will be updated
    '''
    def __init__(self, params: List[Value]) -> None:
        if not all(isinstance(p, Value) for p in params):
            raise TypeError('All parameters must be Value instances')
        self.params = [p for p in params if p.requires_grad]

    def step(self) -> None:
        '''
        Performs a single optimization step to update parameters.
        Step should be called once the gradients are computed.
        Should be overridden by all subclasses'''
        raise NotImplementedError('Subclasses must implement step()')

    def zero_grad(self) -> None:
        '''
        Resets gradients of all parameters to zero.
        Should be called before each backward pass when accumulating
        gradients is not desired
        '''
        for p in self.params:
            p.zero_grad()


class SGD(Optimizer):
    '''
    Implements stochastic gradient descent (optionally with momentum).
    When momentum is used, it maintains a velocity term that accumulates
    gradients over time, creating inertia in parameter updates

    The update rule is:
        grad = grad + weight_decay * param
        velocity = momentum * velocity + (1 - dampening) * grad
        param = param - lr * velocity

    Args:
        params: list of Values to optimize
        lr: learning rate, controls the step size of the optimizer
        momentum: momentum factor, accumulates past gradients
                to smooth updates and make bigger steps
        dampening: reduces the contribution of new gradients to momentum
        weight_decay: adds L2 regularization by penalizing large weights
    '''
    def __init__(
        self,
        params: List[Value],
        lr: float = 1e-3,
        momentum: float = 0,
        dampening: float = 0,
        weight_decay: float = 0
    ) -> None:
        super().__init__(params)

        if lr < 0:
            raise ValueError(f'Learning rate cannot be negative: {lr}')
        if momentum < 0:
            raise ValueError(f'Momentum cannot be negative: {momentum}')
        if not 0 <= dampening <= 1:
            raise ValueError(f'Dampening must be between 0 and 1: {dampening}')
        if weight_decay < 0:
            msg = f'Weight decay cannot be negative: {weight_decay}'
            raise ValueError(msg)

        self.lr: float = lr
        self.momentum: float = momentum
        self.dampening: float = dampening
        self.weight_decay: float = weight_decay
        self.state: Dict[Value, float] = {}

    def clear_state(self) -> None:
        '''Clears the optimizer's momentum buffer state'''
        self.state.clear()

    def step(self) -> None:
        '''
        Performs a single optimization step of SGD.
        For each parameter:
        1. Applies weight decay (if enabled)
        2. Computes momentum update (if enabled)
        3. Updates parameter values using computed gradients
        '''
        for param in self.params:
            grad = param.grad

            # Apply weight decay
            if self.weight_decay != 0:
                grad += self.weight_decay * param.data

            # Momentum update
            if self.momentum != 0:
                if param not in self.state:
                    '''
                    If it's the very first step we set velocity value
                    as current gradient in state dictionary
                    '''
                    self.state[param] = grad
                else:
                    velocity = self.momentum * self.state[param]
                    velocity += (1 - self.dampening) * grad
                    self.state[param] = velocity
                    grad = velocity

            # Update parameter value using the final gradient
            param.data -= self.lr * grad
