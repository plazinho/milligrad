from typing import List, Union, Literal

from .activation import Softmax
from .engine import Value, ValueLike
from .nn import Module


class MSELoss(Module):
    '''
    Creates a criterion that measures the mean squared error
    (squared L2 norm) between each element in the input and target

    Args:
        reduction: specifies the reduction to apply to the output:
            'none': no reduction will be applied
            'mean': the sum of the output will be divided
                    by the number of elements in the output. Set by default
            'sum': the output will be summed
    '''

    def __init__(
        self,
        reduction: Literal['none', 'mean', 'sum'] = 'mean'
    ) -> None:
        self.reduction = reduction

    def forward(
        self,
        prediction: List[Value],
        target: List[ValueLike]
    ) -> Union[Value, List[Value]]:
        if len(prediction) != len(target):
            msg = (f'Size of prediction and target does not match: '
                   f'{len(prediction)} and {len(target)}')
            raise ValueError(msg)
        if self.reduction not in ['none', 'mean', 'sum']:
            msg = 'reduction must be "none", "mean" or "sum"'
            raise ValueError(msg)

        squares = [(x - y)**2 for x, y in zip(prediction, target)]

        if self.reduction == 'none':
            return squares
        if self.reduction == 'sum':
            return sum(squares)  # type: ignore
        return sum(squares)/len(prediction)  # type: ignore


class BCELoss(Module):
    '''
    Creates a criterion that measures the Binary Cross Entropy
    between the target and the input probabilities

    Args:
        reduction: specifies the reduction to apply to the output:
            'none': no reduction will be applied
            'mean': the sum of the output will be divided
                    by the number of elements in the output. Set by default
            'sum': the output will be summed
    '''

    def __init__(
        self,
        reduction: Literal['none', 'mean', 'sum'] = 'mean'
    ) -> None:
        self.reduction = reduction

    def forward(
        self,
        prediction: List[Value],
        target: List[ValueLike]
    ) -> Union[Value, List[Value]]:
        if len(prediction) != len(target):
            msg = (f'Size of prediction and target does not match: '
                   f'{len(prediction)} and {len(target)}')
            raise ValueError(msg)
        if self.reduction not in ['none', 'mean', 'sum']:
            msg = 'reduction must be "none", "mean" or "sum"'
            raise ValueError(msg)

        bce = [
            -(y * x.log() + (1 - y) * (1 - x).log())
            for x, y in zip(prediction, target)
        ]

        if self.reduction == 'none':
            return bce
        if self.reduction == 'sum':
            return sum(bce)  # type: ignore
        return sum(bce)/len(prediction)  # type: ignore


class CrossEntropyLoss(Module):
    '''
    Creates a criterion that computes the cross entropy loss
    between input logits and target.
    Uses softmax internally. Input should be unnormalized logits
    '''

    def forward(
        self,
        prediction: List[Value],
        target: List[ValueLike]
    ) -> Value:
        if len(prediction) != len(target):
            msg = (f'Size of prediction and target does not match: '
                   f'{len(prediction)} and {len(target)}')
            raise ValueError(msg)

        softmax = Softmax()(prediction)

        cel = -sum(
            (y * x.log() for x, y in zip(softmax, target))
        )
        return cel  # type: ignore
