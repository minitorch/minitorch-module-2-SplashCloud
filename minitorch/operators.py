"""Collection of the core mathematical operators used throughout the code base."""

import math
from typing import Callable, Iterable

def mul(a: float, b: float) -> float:
    """Multiple two numbers"""
    return a * b


def id(a: float) -> float:
    """Return input unchanged"""
    return a


def add(a: float, b: float) -> float:
    """Add two numbers"""
    return a + b


def neg(a: float) -> float:
    """Negate a number"""
    return -a


def lt(a: float, b: float) -> float:
    """Check if one number is less than another"""
    return 1.0 if a < b else 0.0


def eq(a: float, b: float) -> float:
    """Check if two numbers are equal"""
    return 1.0 if a == b else 0.0


def max(a: float, b: float) -> float:
    """Return the larger of two numbers"""
    return a if a > b else b


def is_close(a: float, b: float) -> bool:
    """Check if two numbers are close in value"""
    return abs(a - b) < 1e-2


def sigmoid(x: float) -> float:
    """Calculate the sigmoid function"""
    if x >= 0:
        return 1 / (1 + exp(-x))
    return exp(x) / (1 + exp(x))


def relu(x: float) -> float:
    """Apply the ReLu activation function"""
    return max(0.0, x)


def log(x: float) -> float:
    """Calculate the natural logarithm"""
    if x > 0:
        return math.log(x)
    raise ValueError("The domain must be > 0")


def exp(x: float) -> float:
    """Calculate the exponential function"""
    return math.exp(x)


def inv(x: float) -> float:
    """Calculate the reciprocal"""
    if x != 0:
        return 1 / x
    raise ValueError("The domain must be != 0")


def log_back(a: float, b: float) -> float:
    """Compute the derivative of log times a second arg"""
    if a > 0:
        return 1 / a * b
    raise ValueError("The domain must be > 0.")


def inv_back(a: float, b: float) -> float:
    """Compute the derivative of inv times a second arg"""
    if a != 0:
        return -(a ** (-2)) * b
    raise ValueError("The domain must be != 0.")


def relu_back(a: float, b: float) -> float:
    """Compute the derivative of relu times a second arg"""
    if a < 0.0:
        return 0.0
    # the derivative of the relu in x=0 is 1
    return b


def sigmoid_back(a: float, b: float) -> float:
    """Compute the derivative of sigmoid times a second arg"""
    return (sigmoid(a) ** 2) * exp(-a) * b

def map(lst: Iterable[float], func: Callable[[float], float]) -> Iterable[float]:
    """Higher-order function that applies a given function to each element of an iterable"""
    new_lst: Iterable[float] = []
    for item in lst:
        new_lst.append(func(item))
    return new_lst


def zipWith(
    lst1: Iterable[float], lst2: Iterable[float], func: Callable[[float, float], float]
) -> Iterable[float]:
    """Higher-order function that combines elements from two iterables using a given function"""
    new_lst: Iterable[float] = []
    for item1, item2 in zip(lst1, lst2):
        new_lst.append(func(item1, item2))
    return new_lst


def reduce(lst: Iterable[float], func: Callable[[float, float], float]) -> float:
    """Higher-order function that reduces an iterable to a single value using a given function"""
    itr = iter(lst)
    try:
        result = next(itr)  # get the first element
    except StopIteration:
        return 0.0
    for ele in itr:
        result = func(result, ele)
    return result


def negList(lst: Iterable[float]) -> Iterable[float]:
    """Negate all elements in a list"""
    return map(lst, neg)


def addLists(lst1: Iterable[float], lst2: Iterable[float]) -> Iterable[float]:
    """Add corresponding elements from two lists"""
    return zipWith(lst1, lst2, add)


def sum(lst: Iterable[float]) -> float:
    """Sum all elements in a list"""
    return reduce(lst, add)


def prod(lst: Iterable[float]) -> float:
    """Calculate the product of all elements in a list"""
    return reduce(lst, mul)
