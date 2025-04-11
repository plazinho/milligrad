from hypothesis import given, strategies as st
import torch

from milligrad import Value, ReLU, Sigmoid, Tanh, Softmax


tolerance = 1e-4


@given(
    value_1=st.integers(min_value=-50, max_value=50),
    value_2=st.integers(min_value=-50, max_value=50),
    value_3=st.integers(min_value=-50, max_value=50)
)
def test_relu(value_1, value_2, value_3):
    a = [Value(value_1), Value(value_2), Value(value_3)]
    relu = ReLU()
    a_relu_value = relu(a)

    # torch results
    a = torch.tensor([value_1, value_2, value_3]).double()
    relu_torch = torch.nn.ReLU()
    a_relu_torch = relu_torch(a)

    assert all(
        abs(val.data - torch_val.item()) < tolerance
        for val, torch_val in zip(a_relu_value, a_relu_torch)
    )


@given(
    value_1=st.integers(min_value=-50, max_value=50),
    value_2=st.integers(min_value=-50, max_value=50),
    value_3=st.integers(min_value=-50, max_value=50)
)
def test_sigmoid(value_1, value_2, value_3):
    a = [Value(value_1), Value(value_2), Value(value_3)]
    sigmoid = Sigmoid()
    a_sigmoid_value = sigmoid(a)

    # torch results
    a = torch.tensor([value_1, value_2, value_3]).double()
    sigmoid_torch = torch.nn.Sigmoid()
    a_sigmoid_torch = sigmoid_torch(a)

    assert all(
        abs(val.data - torch_val.item()) < tolerance
        for val, torch_val in zip(a_sigmoid_value, a_sigmoid_torch)
    )


@given(
    value_1=st.integers(min_value=-50, max_value=50),
    value_2=st.integers(min_value=-50, max_value=50),
    value_3=st.integers(min_value=-50, max_value=50)
)
def test_tanh(value_1, value_2, value_3):
    a = [Value(value_1), Value(value_2), Value(value_3)]
    tanh = Tanh()
    a_tanh_value = tanh(a)

    # torch results
    a = torch.tensor([value_1, value_2, value_3]).double()
    tanh_torch = torch.nn.Tanh()
    a_tanh_torch = tanh_torch(a)

    assert all(
        abs(val.data - torch_val.item()) < tolerance
        for val, torch_val in zip(a_tanh_value, a_tanh_torch)
    )


@given(
    value_1=st.integers(min_value=-50, max_value=50),
    value_2=st.integers(min_value=-50, max_value=50),
    value_3=st.integers(min_value=-50, max_value=50)
)
def test_softmax(value_1, value_2, value_3):
    a = [Value(value_1), Value(value_2), Value(value_3)]
    softmax = Softmax()
    a_softmax_value = softmax(a)

    # torch results
    a = torch.tensor([value_1, value_2, value_3]).double()
    softmax_torch = torch.nn.Softmax(dim=0)
    a_softmax_torch = softmax_torch(a)

    assert all(
        abs(val.data - torch_val.item()) < tolerance
        for val, torch_val in zip(a_softmax_value, a_softmax_torch)
    )
