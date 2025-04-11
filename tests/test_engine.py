from hypothesis import given, strategies as st
import torch

from milligrad import Value


tolerance = 1e-2


@given(
    value_1=st.one_of(
        st.floats(min_value=-4, max_value=4),
        st.integers(min_value=-4, max_value=4)
        ).filter(lambda x: x < -1e-4 or x > 1e-4),
    value_2=st.one_of(
        st.floats(min_value=0.1, max_value=4),
        st.integers(min_value=-4, max_value=4)
        ).filter(lambda x: x != 0)
)
def test_operations(value_1, value_2):
    a = Value(value_1)
    b = Value(value_2)
    c = (2 * a + a * 2) * a
    d = (2.0 / b + b / 2.0) / b
    e = -(1 + c + c + 1)
    f = 1 - d - d - 1
    z = c - (0 / a + 0 * b + b * 0 - e**2 + f**-2) + d
    z.backward()
    a_value, b_value, z_value = a, b, z

    # torch results
    a = torch.tensor([value_1]).double()
    b = torch.tensor([value_2]).double()
    a.requires_grad = True
    b.requires_grad = True
    c = (2 * a + a * 2) * a
    d = (2.0 / b + b / 2.0) / b
    e = -(1 + c + c + 1)
    f = 1 - d - d - 1
    z = c - (0 / a + 0 * b + b * 0 - e**2 + f**-2) + d
    z.backward()
    a_torch, b_torch, z_torch = a, b, z

    # forward pass
    assert abs(z_value.data - z_torch.item()) < tolerance
    # backward pass
    assert abs(a_value.grad - a_torch.grad.item()) < tolerance
    assert abs(b_value.grad - b_torch.grad.item()) < tolerance


@given(
    value_1=st.one_of(
        st.floats(min_value=-4, max_value=4),
        st.integers(min_value=-4, max_value=4)
        ).filter(lambda x: x < -1e-4 or x > 1e-4),
    value_2=st.one_of(
        st.floats(min_value=0.1, max_value=4),
        st.integers(min_value=1, max_value=4)
        )
)
def test_functions(value_1, value_2):
    a = Value(value_1)
    b = Value(value_2)
    c = (1 + a).exp()
    d = (2 * b).log()
    e = (-2 * c).relu() + d.exp() + d.relu()
    f = (a * b - c * d - e).tanh() + 1.1
    g = 1 / f * 2 + b.sigmoid() - a.sigmoid()
    z = (g**2).log() * (a * b).sigmoid() * (c * d).tanh() + (2 * e * f).relu()
    z.backward()
    a_value, b_value, z_value = a, b, z

    # torch results
    a = torch.tensor([value_1]).double()
    b = torch.tensor([value_2]).double()
    a.requires_grad = True
    b.requires_grad = True
    c = (1 + a).exp()
    d = (2 * b).log()
    e = (-2 * c).relu() + d.exp() + d.relu()
    f = (a * b - c * d - e).tanh() + 1.1
    g = 1 / f * 2 + b.sigmoid() - a.sigmoid()
    z = (g**2).log() * (a * b).sigmoid() * (c * d).tanh() + (2 * e * f).relu()
    z.backward()
    a_torch, b_torch, z_torch = a, b, z

    # forward pass
    assert abs(z_value.data - z_torch.item()) < tolerance
    # backward pass
    assert abs(a_value.grad - a_torch.grad.item()) < tolerance
    assert abs(b_value.grad - b_torch.grad.item()) < tolerance
