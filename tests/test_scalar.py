import pytest

from attention.scalar import Value, neuron


def test_the_neuron_example_has_the_gradients_the_tour_explains():
    v = neuron(0.5, 0.0)
    assert v["L"].data == pytest.approx(4.0)
    v["L"].backward()
    grads = {name: val.grad for name, val in v.items()}
    assert grads == pytest.approx(
        {"L": 1, "e": -4, "p": -4, "w·x": -4, "b": -4, "w": -8, "x": -2, "y": 4}
    )


def test_backward_visits_the_loss_first_and_the_inputs_last():
    v = neuron(0.5, 0.0)
    order = [x.name for x in v["L"].backward_steps()]
    assert order[0] == "L"
    assert order.index("e") < order.index("p") < order.index("w·x") < order.index("w")


def test_gradients_add_up_when_a_value_is_used_twice():
    a = Value(3.0, "a")
    out = a * a
    out.backward()
    assert a.grad == pytest.approx(6.0)


def test_gradient_descent_on_the_neuron_converges():
    w, b = 0.5, 0.0
    for _ in range(30):
        v = neuron(w, b)
        v["L"].backward()
        w -= 0.05 * v["w"].grad
        b -= 0.05 * v["b"].grad
    assert neuron(w, b)["L"].data < 1e-6
