"""Backpropagation on single numbers, small enough to watch every step.

Each Value remembers which values it was made from and how. Calling backward() on the final
one walks the recipe in reverse, handing each input its share of the gradient. That's the
chain rule, and it's exactly what the transformer's backward pass does, just with grids of
numbers instead of single ones.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator


class Value:
    def __init__(self, data: float, name: str = "", parents: tuple[Value, ...] = (), op: str = ""):
        self.data = data
        self.grad = 0.0
        self.name = name
        self.parents = parents
        self.op = op
        self._backward: Callable[[], None] = lambda: None

    def __repr__(self) -> str:
        return f"Value({self.name or self.op}={self.data:.4g}, grad={self.grad:.4g})"

    def __add__(self, other: Value) -> Value:
        out = Value(self.data + other.data, parents=(self, other), op="+")

        def backward() -> None:  # a sum passes its gradient to both inputs unchanged
            self.grad += out.grad
            other.grad += out.grad

        out._backward = backward
        return out

    def __sub__(self, other: Value) -> Value:
        out = Value(self.data - other.data, parents=(self, other), op="−")

        def backward() -> None:
            self.grad += out.grad
            other.grad -= out.grad

        out._backward = backward
        return out

    def __mul__(self, other: Value) -> Value:
        out = Value(self.data * other.data, parents=(self, other), op="×")

        def backward() -> None:  # each input's gradient is the other input times out's
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = backward
        return out

    def square(self) -> Value:
        out = Value(self.data**2, parents=(self,), op="²")

        def backward() -> None:  # the slope of e² is 2e
            self.grad += 2 * self.data * out.grad

        out._backward = backward
        return out

    def order(self) -> list[Value]:
        """Every value this one depends on, each after the values it was made from."""
        seen: set[int] = set()
        out: list[Value] = []

        def visit(v: Value) -> None:
            if id(v) not in seen:
                seen.add(id(v))
                for p in v.parents:
                    visit(p)
                out.append(v)

        visit(self)
        return out

    def backward_steps(self) -> Iterator[Value]:
        """Backpropagate one value at a time, from this one back to the inputs."""
        for v in self.order():
            v.grad = 0.0
        self.grad = 1.0
        for v in reversed(self.order()):
            v._backward()
            yield v

    def backward(self) -> None:
        for _ in self.backward_steps():
            pass


def neuron(w: float, b: float, x: float = 2.0, y: float = 3.0) -> dict[str, Value]:
    """The smallest learning problem: predict y from x with p = w·x + b, loss (p − y)²."""
    vw, vx, vb, vy = Value(w, "w"), Value(x, "x"), Value(b, "b"), Value(y, "y")
    m = vw * vx
    m.name = "w·x"
    p = m + vb
    p.name = "p"
    e = p - vy
    e.name = "e"
    loss = e.square()
    loss.name = "L"
    return {"w": vw, "x": vx, "b": vb, "y": vy, "w·x": m, "p": p, "e": e, "L": loss}
