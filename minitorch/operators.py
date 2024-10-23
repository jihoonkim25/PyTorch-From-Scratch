"""
Collection of the core mathematical operators used throughout the code base.
"""

import math
from typing import Callable, Iterable

# ## Task 0.1
#
# Implementation of a prelude of elementary functions.


def mul(x: float, y: float) -> float:
    "$f(x, y) = x * y$"
    return x * y


def id(x: float) -> float:
    "$f(x) = x$"
    return x


def add(x: float, y: float) -> float:
    "$f(x, y) = x + y$"
    return x + y


def neg(x: float) -> float:
    "$f(x) = -x$"
    return -1.0 * x


def lt(x: float, y: float) -> float:
    "$f(x) =$ 1.0 if x is less than y else 0.0"
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    "$f(x) =$ 1.0 if x is equal to y else 0.0"
    return float(int(x == y))


def max(x: float, y: float) -> float:
    "$f(x) =$ x if x is greater than y else y"
    return x if x > y else y


def is_close(x: float, y: float) -> float:
    "$f(x) = |x - y| < 1e-2$"
    return 1.0 if abs(x - y) < 1e-2 else 0.0


def sigmoid(x: float) -> float:
    r"""
    $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$

    (See https://en.wikipedia.org/wiki/Sigmoid_function )

    Calculate as

    $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$

    for stability.
    """
    return (
        (1.0 / (1.0 + math.exp(-x))) if x >= 0 else (math.exp(x) / (1.0 + math.exp(x)))
    )


def relu(x: float) -> float:
    """
    $f(x) =$ x if x is greater than 0, else 0

    (See https://en.wikipedia.org/wiki/Rectifier_(neural_networks) .)
    """
    return x if x > 0.0 else 0.0


EPS = 1e-6


def log(x: float) -> float:
    "$f(x) = log(x)$"
    return math.log(x + EPS)


def exp(x: float) -> float:
    "$f(x) = e^{x}$"
    return math.exp(x)


def log_back(x: float, d: float) -> float:
    r"If $f = log$ as above, compute $d \times f'(x)$"
    return d * (1.0 / (x + EPS))


def inv(x: float) -> float:
    "$f(x) = 1/x$"
    if x == 0:
        raise ValueError("Cannot divide by 0!")
    return 1.0 / x


def inv_back(x: float, d: float) -> float:
    "If $f(x) = 1/x$ compute $d \times f'(x)$"
    return d * (-1.0 / (x**2))


def relu_back(x: float, d: float) -> float:
    r"If $f = relu$ compute $d \times f'(x)$"
    return d * (1.0 if x > 0.0 else 0.0)


# ## Task 0.3

# Small practice library of elementary higher-order functions.


def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """
    Higher-order map.

    See https://en.wikipedia.org/wiki/Map_(higher-order_function)

    Args:
        fn (Callable[[float], float]): Function from one value to one value.

    Returns:
        Callable[[Iterable[float]], Iterable[float]]: A function that takes a list, applies `fn` to each element, and returns a new list
    """

    def map_to_iterable(iterable: Iterable[float]) -> Iterable[float]:
        """
        Map a function to each value in an iterable

        Args:
            iterable (Iterable[float]): Iterable to map to

        Returns:
            Iterable[float]: new list with each element of iterable mapped to `fn`
        """
        return [fn(i) for i in iterable]

    return map_to_iterable


def negList(ls: Iterable[float]) -> Iterable[float]:
    """
    Use `map` and `neg` to negate each element in `ls`

    Args:
        ls (Iterable[float]): A list.

    Returns:
        Iterable[float]: A function that takes a list and negates each element, then returns the
        negated list
    """
    negate = map(lambda x: -x)
    return negate(ls)


def zipWith(
    fn: Callable[[float, float], float]
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """
    Higher-order zipwith (or map2).

    See https://en.wikipedia.org/wiki/Map_(higher-order_function)

    Args:
        fn: combine two values

    Returns:
        Function that takes two equally sized lists `ls1` and `ls2`, produce a new list by
         applying fn(x, y) on each pair of elements.

    """

    def zip_to_iterable(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
        """
        Take two equally sized lists and apply fn() on each pair of elements

        Args:
            ls1 (Iterable[float]): first list
            ls2 (Iterable[float]): second list

        Returns:
            Iterable[float]: zipped list of ls1 and ls2 with fn applied to each pair of elements.
        """
        return [fn(i, j) for i, j in zip(ls1, ls2)]

    return zip_to_iterable


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    """
    Add the elements of `ls1` and `ls2` using `zipWith` and `add`

    Args:
        ls1 (Iterable[float]): first list to add
        ls2 (Iterable[float]): second list to add

    Returns:
        Iterable[float]: Function that takes two equally sized lists 'ls1' and 'ls2' and produces a new list
        by adding each pair of elements.
    """
    zip_add_fn = zipWith(add)
    return zip_add_fn(ls1, ls2)


def reduce(
    fn: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    r"""
    Higher-order reduce.

    Args:
        fn (Callable[[float, float], float]): combine two values
        start (float): start value $x_0$

    Returns:
        Callable[Iterable[float], float]: Function that takes a list `ls` of elements
         $x_1 \ldots x_n$ and computes the reduction :math:`fn(x_3, fn(x_2,
         fn(x_1, x_0)))`
    """

    def reduce_list(ls: Iterable[float]) -> float:
        """
        Compute the reduction given a list with fn()

        Args:
            ls (Iterable[float]): List to compute reduction on

        Returns:
            float: reduced value
        """
        res = start
        for i in ls:
            res = fn(res, i)
        return res

    return reduce_list


def sum(ls: Iterable[float]) -> float:
    """
    Sum up a list using `reduce` and `add`.

    Args:
        ls (Iterable[float]): List to sum up

    Returns:
        float: The sum of all elements in the list.
    """
    reduce_add = reduce(add, 0)
    return reduce_add(ls)


def prod(ls: Iterable[float]) -> float:
    """
    Product of a list using `reduce` and `mul`.

    Args:
        ls (Iterable[float]): List to multiply up

    Returns:
        float: The product of all elements in the list.
    """
    reduce_mul = reduce(mul, 1)
    return reduce_mul(ls)