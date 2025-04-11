# milligrad

**milligrad** is an automatic differentiation engine for scalars, which also includes a small NN library with PyTorch-like usage. Built entirely for educational purposes in pure Python: no `numpy`, nothing.

Mainly insipred by Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd) and Cornell's Tech [MiniTorch](https://minitorch.github.io)

## Features

- **Automatic Differentiation**: The core `Value` class supports automatic differentiation for scalar values, enabling easy computation of gradients.
- **Neural Network Modules**: Provides PyTorch-like modules such as `Module`, `Linear` and `Sequential` for building neural networks.
- **Activation Functions**: Includes common activation functions like `ReLU`, `Sigmoid`, `Tanh`, and `Softmax`.
- **Loss Functions**: Implements loss functions such as `MSELoss`, `BCELoss`, and `CrossEntropyLoss`.
- **Optimizers**: Offers an `SGD` optimizer with support for momentum, weight decay, and dampening.

As well as the `tests/` directory, which contains several tests that compare the engine's functionality with PyTorch's implementation, ensuring correctness and compatibility.

## Installation

To install and use milligrad you just need to clone the repository:

```bash
git clone https://github.com/plazinho/milligrad.git
```

If you want to run the tests as well, you need to install the required libraries:

```bash
git clone https://github.com/plazinho/milligrad.git
cd milligrad
pip install -r requirements.txt
```

And then to run the test:

```bash
pytest tests
```

## Usage

### Notebook example

Check out the `digits.ipynb` notebook where you train a NN for a classification task similar to MNIST. Unfortunately, the MNIST dataset is too heavy for a scalar engine like this one.

### Basic example

An example showing some of the supported operations and how to call backpropagation.

```python
from milligrad import Value

a = Value(1)
b = Value(-2)
c = (2 * a + 2 * b) * a - b
d = a * b - c**2 + c.relu()
z = d.exp() - (c**2).log() + (a * b).sigmoid() + (c * d).tanh()
# backpropagation. computes the gradients of z with respect to a, b, c, and d
z.backward()

print(f'Result of the forward pass: {z.data:.4f}')
print(f'Gradient of a: {a.grad:.4f}')
print(f'Gradient of b: {b.grad:.4f}')
print(f'Gradient of c: {c.grad:.4f}')
print(f'Gradient of d: {d.grad:.4f}')

>>> Result of the forward pass: 18.6752
>>> Gradient of a: -0.4807
>>> Gradient of b: -1.7597
>>> Gradient of c: -2.0000
>>> Gradient of d: 0.1353
```

### Neural Network example with XOR problem

It's still may not converge depending on your NN's architecture, weight initialization, optimizer hyperparameters, etc.

```python
from milligrad import Module, Linear, Sequential, ReLU, SGD, Sigmoid, BCELoss

data = [
  [0, 0],
  [1, 1],
  [1, 0],
  [0, 1],
]
target = [0, 0, 1, 1]


# Define a neural network
class XORModel(Module):
    def __init__(self):
        super().__init__()
        self.sequential = Sequential(
            Linear(2, 4),
            ReLU(),
            Linear(4, 1),
            Sigmoid(),
        )

    def forward(self, x):
        return self.sequential(x)


# Initialize your model
model = XORModel()

# Define a loss function
loss_fn = BCELoss()

# Create an optimizer
optimizer = SGD(model.parameters(), lr=0.5, momentum=0.9)

# Training loop
for epoch in range(100):
    preds = []
    for x in data:
        output = model(x)
        preds.append(output[0])

    loss = loss_fn(preds, target)
    if loss.data < 0.01:
        print('Converged')
        break

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f'predictions: {[round(val.data, 4) for val in preds]}')
print(f'target: {target}')
print(f'final loss: {loss.data:.4f}')

>>> Converged
>>> predictions: [0.0306, 0.0034, 0.9956, 0.9998]
>>> target: [0, 0, 1, 1]
>>> final loss: 0.0098
```
