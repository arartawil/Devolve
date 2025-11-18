"""Adaptive DE algorithm implementations."""

from .jde import JDE
from .jade import JADE
from .shade import SHADE
from .lshade import LSHADE
from .lshade_epsin import LSHADEEpSin

__all__ = ['JDE', 'SaDE', 'JADE', 'SHADE', 'LSHADE', 'LSHADEEpSin', 'LSHADEcnEpSin']

# Stub classes - to be implemented

class SaDE:
    """SaDE - Self-adaptive Differential Evolution (Qin et al., 2009)"""
    pass

class LSHADEcnEpSin:
    """LSHADE-cnEpSin - Latest state-of-the-art (Kumar et al., 2021)"""
    pass
