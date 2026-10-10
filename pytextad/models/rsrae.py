"""Moved to :mod:`pytextad.models.vector.rsrae`; this alias keeps old imports working."""
import sys

from .vector import rsrae as _rsrae

sys.modules[__name__] = _rsrae
