from hypothesis import given, strategies as st
import torch

from milligrad import Value, MSELoss, BCELoss, CrossEntropyLoss


tolerance = 1e-4


@given(
    x1=st.integers(min_value=-50, max_value=50),
    x2=st.integers(min_value=-50, max_value=50),
    x3=st.integers(min_value=-50, max_value=50),
    y1=st.integers(min_value=-50, max_value=50),
    y2=st.integers(min_value=-50, max_value=50),
    y3=st.integers(min_value=-50, max_value=50)
)
def test_mse_loss(x1, x2, x3, y1, y2, y3):
    prediction = [Value(x1), Value(x2), Value(x3)]
    target = [y1, y2, y3]
    loss_fn_value = MSELoss()
    loss_value = loss_fn_value(prediction, target)

    # torch results
    prediction_torch = torch.tensor([x1, x2, x3]).double()
    target_torch = torch.tensor([y1, y2, y3]).double()
    loss_fn_torch = torch.nn.MSELoss()
    loss_torch = loss_fn_torch(prediction_torch, target_torch)

    assert abs(loss_value.data - loss_torch.item()) < tolerance


@given(
    x1=st.floats(min_value=0, max_value=1).filter(
        lambda x: 1e-5 < x < 1 - 1e-5
    ),
    x2=st.floats(min_value=0, max_value=1).filter(
        lambda x: 1e-5 < x < 1 - 1e-5
    ),
    x3=st.floats(min_value=0, max_value=1).filter(
        lambda x: 1e-5 < x < 1 - 1e-5
    ),
    y1=st.floats(min_value=0, max_value=1),
    y2=st.floats(min_value=0, max_value=1),
    y3=st.floats(min_value=0, max_value=1)
)
def test_bce_loss(x1, x2, x3, y1, y2, y3):
    prediction = [Value(x1), Value(x2), Value(x3)]
    target = [y1, y2, y3]
    loss_fn_value = BCELoss()
    loss_value = loss_fn_value(prediction, target)

    # torch results
    prediction_torch = torch.tensor([x1, x2, x3]).double()
    target_torch = torch.tensor([y1, y2, y3]).double()
    loss_fn_torch = torch.nn.BCELoss()
    loss_torch = loss_fn_torch(prediction_torch, target_torch)

    assert abs(loss_value.data - loss_torch.item()) < tolerance


@given(
    x1=st.integers(min_value=-50, max_value=50),
    x2=st.integers(min_value=-50, max_value=50),
    x3=st.integers(min_value=-50, max_value=50),
    y1=st.floats(min_value=0, max_value=1),
    y2=st.floats(min_value=0, max_value=1),
    y3=st.floats(min_value=0, max_value=1)
)
def test_cross_entropy_loss(x1, x2, x3, y1, y2, y3):
    prediction = [Value(x1), Value(x2), Value(x3)]
    target = [y1, y2, y3]
    loss_fn_value = CrossEntropyLoss()
    loss_value = loss_fn_value(prediction, target)

    # torch results
    prediction_torch = torch.tensor([x1, x2, x3]).double()
    target_torch = torch.tensor([y1, y2, y3]).double()
    loss_fn_torch = torch.nn.CrossEntropyLoss()
    loss_torch = loss_fn_torch(prediction_torch, target_torch)

    assert abs(loss_value.data - loss_torch.item()) < tolerance
