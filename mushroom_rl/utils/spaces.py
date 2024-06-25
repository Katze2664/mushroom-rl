import numpy as np
from numpy.typing import NDArray
from typing import Any

from mushroom_rl.core import Serializable

def short_repr(arr: NDArray[Any]) -> str:  # Copied from gymnasium.spaces.box
    """Create a shortened string representation of a numpy array.

    If arr is a multiple of the all-ones vector, return a string representation of the multiplier.
    Otherwise, return a string representation of the entire array.

    Args:
        arr: The array to represent

    Returns:
        A short representation of the array
    """
    if arr.size != 0 and np.min(arr) == np.max(arr):
        return str(np.min(arr))
    return str(arr)

class Box(Serializable):
    """
    This class implements functions to manage continuous states and action
    spaces. It is similar to the ``Box`` class in ``gym.spaces.box``.

    """
    def __init__(self, low, high, shape=None):
        """
        Constructor.

        Args:
            low ([float, np.ndarray]): the minimum value of each dimension of
                the space. If a scalar value is provided, this value is
                considered as the minimum one for each dimension. If a
                np.ndarray is provided, each i-th element is considered the
                minimum value of the i-th dimension;
            high ([float, np.ndarray]): the maximum value of dimensions of the
                space. If a scalar value is provided, this value is considered
                as the maximum one for each dimension. If a np.ndarray is
                provided, each i-th element is considered the maximum value
                of the i-th dimension;
            shape (np.ndarray, None): the dimension of the space. Must match
                the shape of ``low`` and ``high``, if they are np.ndarray.

        """
        if shape is None:
            self._low = low
            self._high = high
            self._shape = low.shape
        else:
            self._low = low
            self._high = high
            self._shape = shape
            if np.isscalar(low) and np.isscalar(high):
                self._low += np.zeros(shape)
                self._high += np.zeros(shape)

        assert self._low.shape == self._high.shape

        self._add_save_attr(
            _low='numpy',
            _high='numpy'
        )

    @property
    def low(self):
        """
        Returns:
             The minimum value of each dimension of the space.

        """
        return self._low

    @property
    def high(self):
        """
        Returns:
             The maximum value of each dimension of the space.

        """
        return self._high

    @property
    def shape(self):
        """
        Returns:
            The dimensions of the space.

        """
        return self._shape

    def _post_load(self):
        self._shape = self._low.shape
    
    def __repr__(self) -> str:  # Modified from gymnasium.spaces.box
        """A string representation of this space.

        The representation will include bounds and shape.
        If a bound is uniform, only the corresponding scalar will be given to avoid redundant and ugly strings.

        Returns:
            A representation of the space
        """
        return f"Box(low={short_repr(self.low)}, high={short_repr(self.high)}, shape={self.shape})"


class Discrete(Serializable):
    """
    This class implements functions to manage discrete states and action
    spaces. It is similar to the ``Discrete`` class in ``gym.spaces.discrete``.

    """
    def __init__(self, n):
        """
        Constructor.

        Args:
            n (int): the number of values of the space.

        """
        self.values = np.arange(n)
        self.n = n

        self._add_save_attr(
            n='primitive',
            values='numpy'
        )

    @property
    def size(self):
        """
        Returns:
            The number of elements of the space.

        """
        return self.n,

    @property
    def shape(self):
        """
        Returns:
            The shape of the space that is always (1,).

        """
        return 1,

    def __repr__(self) -> str:  # Modified from gymnasium.spaces.discrete
        """Gives a string representation of this space."""
        return f"Discrete(n={self.n})"
