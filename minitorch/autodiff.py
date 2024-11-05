from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    def add_delta(val: Any, delta: float) -> Any:
        return val + delta
    
    args1 = list(vals)
    args2 = list(vals)
    args1[arg] = add_delta(args1[arg], epsilon/2)
    args2[arg] = add_delta(args2[arg], -epsilon/2)
    return (f(*args1) - f(*args2)) / epsilon


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    # # # # # # # # # # # # # # # # # # # BUG # # # # # # # # # # # # # # # # # # # # #
    # Need to use DFS to implement the topological order, instead of BFS              #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    res = []
    visited = set()

    def dfs(node: Variable):
        if node.is_constant():
            return
        if node.unique_id in visited:
            return
        for parent in node.parents:
            dfs(parent)
        visited.add(node.unique_id)
        res.append(node)
    
    dfs(variable)
    return list(reversed(res))


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """

    # 1. get the topological node sequence
    sequence = topological_sort(variable)
    out = sequence[0]
    dict = {out.unique_id: [deriv]}
    for node in sequence:
        if node.is_leaf():
            node.accumulate_derivative(sum(dict[node.unique_id]))
            continue
        pairs = node.chain_rule(sum(dict[node.unique_id]))
        for pair in pairs:
            if pair[0].unique_id not in dict.keys():
                dict[pair[0].unique_id] = []
            dict[pair[0].unique_id].append(pair[1])

@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
