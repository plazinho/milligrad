from hypothesis import given, strategies as st
import torch
import torch.nn as nn

from milligrad import Value, Parameter, Module, Linear, ReLU, Tanh, \
    Sequential, Sigmoid


tolerance = 1e-6
# to make it easier we set weights and biases shapes
in_features = 4
out_features = 8
output_size = 4


@given(
    inputs=st.lists(st.floats(min_value=-10, max_value=10),
                    min_size=in_features,
                    max_size=in_features),
    weights_1=st.lists(st.floats(min_value=-1, max_value=1),
                       min_size=in_features * out_features,
                       max_size=in_features * out_features),
    weights_2=st.lists(st.floats(min_value=-1, max_value=1),
                       min_size=out_features * output_size,
                       max_size=out_features * output_size),
    biases_1=st.lists(st.floats(min_value=-1, max_value=1),
                      min_size=out_features,
                      max_size=out_features),
    biases_2=st.lists(st.floats(min_value=-1, max_value=1),
                      min_size=output_size,
                      max_size=output_size)
)
def test_linear(inputs, weights_1, weights_2, biases_1, biases_2):

    class MyNN(Module):
        def __init__(self):
            super().__init__()
            self.linear_1 = Linear(in_features, out_features)
            self.relu = ReLU()
            self.linear_2 = Linear(out_features, output_size)
            self.tanh = Tanh()

        def forward(self, x):
            out = self.linear_1(x)
            out = self.relu(out)
            out = self.linear_2(out)
            out = self.tanh(out)
            return out

    my_model = MyNN()
    # set generated weights and biases
    my_model.linear_1.weights = Parameter([Value(val) for val in weights_1])
    my_model.linear_1.biases = Parameter([Value(val) for val in biases_1])
    my_model.linear_2.weights = Parameter([Value(val) for val in weights_2])
    my_model.linear_2.biases = Parameter([Value(val) for val in biases_2])

    # forward pass
    my_model_output = my_model(inputs)

    # torch model
    class TorchNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_1 = nn.Linear(in_features, out_features)
            self.relu = nn.ReLU()
            self.linear_2 = nn.Linear(out_features, output_size)
            self.tanh = nn.Tanh()

        def forward(self, x):
            out = self.linear_1(x)
            out = self.relu(out)
            out = self.linear_2(out)
            out = self.tanh(out)
            return out

    torch_model = TorchNN()
    # since torch operates on tensors
    # we need to reshape generated weights and biases
    torch_w1 = torch.tensor(weights_1).double().reshape(out_features, in_features)
    torch_b1 = torch.tensor(biases_1).double().reshape(out_features)
    torch_w2 = torch.tensor(weights_2).double().reshape(output_size, out_features)
    torch_b2 = torch.tensor(biases_2).double().reshape(output_size)
    torch_model.linear_1.weight = nn.Parameter(torch_w1)
    torch_model.linear_1.bias = nn.Parameter(torch_b1)
    torch_model.linear_2.weight = nn.Parameter(torch_w2)
    torch_model.linear_2.bias = nn.Parameter(torch_b2)

    # forward pass
    model_output_torch = torch_model(torch.tensor(inputs).double())

    # sum the outputs of the models
    my_model_sum = sum((val.data for val in my_model_output))
    torch_model_sum = model_output_torch.sum().item()

    # assert the sum of the outputs of the models are close
    assert abs(my_model_sum - torch_model_sum) < tolerance


@given(
    inputs=st.lists(st.floats(min_value=-10, max_value=10),
                    min_size=in_features,
                    max_size=in_features),
    weights_1=st.lists(st.floats(min_value=-1, max_value=1),
                       min_size=in_features * out_features,
                       max_size=in_features * out_features),
    weights_2=st.lists(st.floats(min_value=-1, max_value=1),
                       min_size=out_features * output_size,
                       max_size=out_features * output_size),
    biases_1=st.lists(st.floats(min_value=-1, max_value=1),
                      min_size=out_features,
                      max_size=out_features),
    biases_2=st.lists(st.floats(min_value=-1, max_value=1),
                      min_size=output_size,
                      max_size=output_size)
)
def test_sequential(inputs, weights_1, weights_2, biases_1, biases_2):

    class MyNN(Module):
        def __init__(self):
            super().__init__()
            self.sequential = Sequential(
                Linear(in_features, out_features),
                ReLU()
            )
            self.linear = Linear(out_features, output_size)
            self.sigmoid = Sigmoid()

        def forward(self, x):
            out = self.sequential(x)
            out = self.linear(out)
            out = self.sigmoid(out)
            return out

    my_model = MyNN()
    # set generated weights and biases
    my_model.sequential.modules()[0].weights = Parameter([Value(val) for val in weights_1])
    my_model.sequential.modules()[0].biases = Parameter([Value(val) for val in biases_1])
    my_model.linear.weights = Parameter([Value(val) for val in weights_2])
    my_model.linear.biases = Parameter([Value(val) for val in biases_2])

    # forward pass
    my_model_output = my_model(inputs)

    # torch model
    class TorchNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.sequential = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.ReLU()
            )
            self.linear = nn.Linear(out_features, output_size)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            out = self.sequential(x)
            out = self.linear(out)
            out = self.sigmoid(out)
            return out

    torch_model = TorchNN()
    # since torch operates on tensors
    # we need to reshape generated weights and biases
    torch_w1 = torch.tensor(weights_1).double().reshape(out_features, in_features)
    torch_b1 = torch.tensor(biases_1).double().reshape(out_features)
    torch_w2 = torch.tensor(weights_2).double().reshape(output_size, out_features)
    torch_b2 = torch.tensor(biases_2).double().reshape(output_size)
    torch_model.sequential[0].weight = nn.Parameter(torch_w1)
    torch_model.sequential[0].bias = nn.Parameter(torch_b1)
    torch_model.linear.weight = nn.Parameter(torch_w2)
    torch_model.linear.bias = nn.Parameter(torch_b2)

    # forward pass
    model_output_torch = torch_model(torch.tensor(inputs).double())

    # sum the outputs of the models
    my_model_sum = sum((val.data for val in my_model_output))
    torch_model_sum = model_output_torch.sum().item()

    # assert the sum of the outputs of the models are close
    assert abs(my_model_sum - torch_model_sum) < tolerance
