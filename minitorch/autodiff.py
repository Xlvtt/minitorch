from dataclasses import dataclass
from typing import Any, Iterable, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
    -------
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    plus = list(vals)
    minus = list(vals)
    plus[arg] += epsilon
    minus[arg] -= epsilon
    return (f(*plus) - f(*minus)) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None: ...

    @property
    def unique_id(self) -> int: ...

    def is_leaf(self) -> bool: ...

    def is_constant(self) -> bool: ...

    @property
    def parents(self) -> Iterable["Variable"]: ...

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]: ...


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
    ----
        variable: The right-most variable

    Returns:
    -------
        Non-constant Variables in topological order starting from the right.

    """
    visited = set()
    order = []

    def visit(node: Variable) -> None:
        if node.is_constant() or node.unique_id in visited:
            return
        visited.add(node.unique_id)
        for parent in node.parents:
            visit(parent)
        order.append(node)

    visit(variable)
    return reversed(order)


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
    ----
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.

    """
    derivatives = {variable.unique_id: deriv}
    for node in topological_sort(variable):
        d_node = derivatives[node.unique_id]
        if node.is_leaf():
            node.accumulate_derivative(d_node)
        else:
            for parent, d_parent in node.chain_rule(d_node):
                derivatives[parent.unique_id] = (
                    derivatives.get(parent.unique_id, 0) + d_parent
                )


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
