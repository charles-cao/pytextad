"""Moved to :mod:`pytextad.models.embedding.rsrae`; this alias keeps old imports working."""
import sys

from .embedding import rsrae as _rsrae

sys.modules[__name__] = _rsrae
