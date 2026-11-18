"""Turn DNA sequences into numeric k-mer frequency features."""
import itertools

import numpy as np


def kmer_names(k=4):
    """All 4**k possible k-mers in a fixed order: AAAA, AAAC, AAAG, ..."""
    return ["".join(p) for p in itertools.product("ACGT", repeat=k)]


def kmer_frequency_matrix(sequences, k=4):
    """Return (X, names) where X[i, j] is the frequency of k-mer j in sequence i.

    Each row sums to 1, so sequences of slightly different lengths stay comparable.
    """
    names = kmer_names(k)
    index = {kmer: col for col, kmer in enumerate(names)}
    counts = np.zeros((len(sequences), len(names)))
    for row, seq in enumerate(sequences):
        for start in range(len(seq) - k + 1):
            col = index.get(seq[start:start + k])
            if col is not None:  # skip any k-mer containing a non-ACGT character
                counts[row, col] += 1
    totals = counts.sum(axis=1, keepdims=True)
    if (totals == 0).any():
        raise ValueError("Some sequences contain no valid k-mers.")
    return counts / totals, names


def gc_content(sequences):
    """Fraction of G and C bases in each sequence (a simple interpretable property)."""
    return np.array([(s.count("G") + s.count("C")) / len(s) for s in sequences])