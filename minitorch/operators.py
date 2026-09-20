"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


def mul(x: float, y: float) -> float:
    """Multiply two numbers."""
    return x * y


def id(x: float) -> float:
    """Return the input unchanged."""
    return x


def add(x: float, y: float) -> float:
    """Add two numbers."""
    return x + y


def neg(x: float) -> float:
    """Negate a number."""
    return -x


def lt(x: float, y: float) -> float:
    """Return one when x is less than y, otherwise zero."""
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    """Return one when x equals y, otherwise zero."""
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    """Return the larger of two numbers."""
    return x if x > y else y


def is_close(x: float, y: float) -> bool:
    """Check whether two numbers differ by less than 0.01."""
    return abs(x - y) < 1e-2


def sigmoid(x: float) -> float:
    """Compute the logistic sigmoid of x."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Return x when positive, otherwise zero."""
    return max(0.0, x)


def log(x: float) -> float:
    """Compute the natural logarithm of x."""
    return math.log(x)


def exp(x: float) -> float:
    """Compute the exponential of x."""
    return math.exp(x)


def inv(x: float) -> float:
    """Compute the reciprocal of x."""
    return 1.0 / x


def log_back(x: float, d: float) -> float:
    """Backpropagate d through the natural logarithm."""
    return d * (1.0 / x)


def inv_back(x: float, d: float) -> float:
    """Backpropagate d through the reciprocal."""
    return -d * (1.0 / x**2)


def relu_back(x: float, d: float) -> float:
    """Backpropagate d through ReLU."""
    return 0 if x <= 0 else d


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


def map(fn: Callable[[float], float], ls: Iterable[float]) -> list[float]:
    """Apply fn to every element of ls."""
    return [fn(x) for x in ls]


def zipWith(
    fn: Callable[[float, float], float], it1: Iterable[float], it2: Iterable[float]
) -> list[float]:
    """Apply fn to corresponding elements of two iterables."""
    return [fn(x, y) for x, y in zip(it1, it2)]


def reduce(
    fn: Callable[[float, float], float], ls: Iterable[float], init: float
) -> float:
    """Fold ls from left to right, starting at init."""
    result = init
    for el in ls:
        result = fn(result, el)
    return result


def negList(ls: Iterable[float]) -> list[float]:
    """Negate each element of ls."""
    return map(neg, ls)


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> list[float]:
    """Add corresponding elements of two lists."""
    return zipWith(add, ls1, ls2)


def sum(ls: Iterable[float]) -> float:
    """Sum all elements of ls."""
    return reduce(add, ls, 0.0)


def prod(ls: Iterable[float]) -> float:
    """Multiply all elements of ls."""
    return reduce(mul, ls, 1.0)
