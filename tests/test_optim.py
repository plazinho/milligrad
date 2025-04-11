from hypothesis import given, settings, strategies as st
import torch

from milligrad import Value, SGD


tolerance = 1e-3


@settings(deadline=None)
@given(
    w1=st.floats(min_value=-1, max_value=1),
    w2=st.floats(min_value=-1, max_value=1),
    w3=st.floats(min_value=-1, max_value=1),
    lr=st.floats(min_value=1e-6, max_value=1),
    momentum=st.floats(min_value=0, max_value=1),
    weight_decay=st.floats(min_value=0, max_value=1),
    dampening=st.floats(min_value=0, max_value=1),
    epochs=st.integers(min_value=1, max_value=5)
)
def test_optim(w1, w2, w3, lr, momentum, weight_decay, dampening, epochs):

    def dummy_loss(a, b, c):
        return a**2 + b**2 + c**2 - a * b * c

    a = Value(w1)
    b = Value(w2)
    c = Value(w3)

    optimizer = SGD(
        [a, b, c],
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        dampening=dampening
    )

    for _ in range(epochs):
        optimizer.zero_grad()
        loss = dummy_loss(a, b, c)
        loss.backward()
        optimizer.step()

    loss_value = loss.data
    a_value = a.data
    b_value = b.data
    c_value = c.data

    # torch results
    a = torch.tensor([w1])
    b = torch.tensor([w2])
    c = torch.tensor([w3])
    a.requires_grad = True
    b.requires_grad = True
    c.requires_grad = True

    optimizer = torch.optim.SGD(
        [a, b, c],
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        dampening=dampening
    )

    for _ in range(epochs):
        optimizer.zero_grad()
        loss = dummy_loss(a, b, c)
        loss.backward()
        optimizer.step()

    loss_torch = loss.item()
    a_torch = a.item()
    b_torch = b.item()
    c_torch = c.item()

    assert abs(loss_value - loss_torch) < tolerance
    assert abs(a_value - a_torch) < tolerance
    assert abs(b_value - b_torch) < tolerance
    assert abs(c_value - c_torch) < tolerance
