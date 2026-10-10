"""TokenCore (Cao et al., WWW 2026)."""
from ._vendor.tokencore import TokenCore as _TokenCore
from .base import BaseEmbeddingDetector


class TokenCore(BaseEmbeddingDetector):
    """Nearest-neighbour memory bank of normal token embeddings :cite:`cao2026tokencore`.

    Every training vector is stored; a vector's anomaly score is its distance to the
    nearest training vectors. Used on word vectors pooled from sub-words
    (``TokenEmbedder(..., word_pooling="max")``) through
    :class:`~pytextad.models.wrappers.TokenDetector`, it is the TokenCore method as published;
    with sub-word vectors (the default), sub-words are scored and their scores combined.

    Parameters
    ----------
    n_neighbors : int, default=1
        Number of nearest training vectors.
    aggregation : {"max", "mean", "min", "median"}, default="max"
        How the ``n_neighbors`` distances are combined (only used if ``n_neighbors > 1``).
        This is not the token-to-document aggregation of ``TokenDetector``.
    contamination : float, default=0.1
        Expected proportion of anomalies; sets ``threshold_`` for ``predict``.
    random_state : int or None, default=None
        TokenCore is deterministic; accepted for a uniform interface.
    verbose : bool, default=False
        Unused.

    Notes
    -----
    Code: the authors' implementation (Yang Cao), unchanged.

    Examples
    --------
    >>> from pytextad import TokenCore, TokenDetector, TokenEmbedder
    >>> det = TokenDetector(TokenCore(), embedder=TokenEmbedder("bert-base-uncased"))
    >>> word_scores = det.fit(train_words).token_scores(test_words)
    """

    def __init__(self, n_neighbors=1, aggregation="max", contamination=0.1, random_state=None,
                 verbose=False):
        super().__init__(contamination, random_state, device="cpu", verbose=verbose)
        self.n_neighbors = n_neighbors
        self.aggregation = aggregation

    def _build(self, n_features):
        return _TokenCore(n_neighbors=self.n_neighbors, aggregation=self.aggregation)
