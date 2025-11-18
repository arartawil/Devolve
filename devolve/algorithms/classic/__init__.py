"""Classic Differential Evolution variants."""

from .derand1 import DErand1
from .debest1 import DEbest1
from .decurrent_to_best1 import DEcurrentToBest1
from .derand2 import DErand2
from .debest2 import DEbest2

__all__ = [
    'DErand1',
    'DEbest1',
    'DEcurrentToBest1',
    'DErand2',
    'DEbest2',
]
