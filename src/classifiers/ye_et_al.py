"""[Optimizing F-measures: A Tale of Two Approaches] - https://arxiv.org/pdf/1206.4625.pdf
Algorithm 1.

Implementation entirely based on that from Arkadiusz Jachnik
https://github.com/arkadiusz-jachnik/MLC-PCC/blob/master/mlc_pcc/src/put/mlc/classifiers/f/QuadraticNaiveFMaximizer.java
"""

import numpy as np
from tqdm import tqdm


class QuadraticTimeAlgorithm(object):
    """[Optimizing F-measures: A Tale of Two Approaches] - https://arxiv.org/pdf/1206.4625.pdf
    Algorithm 1.

    Implementation entirely based on that from Arkadiusz Jachnik
    https://github.com/arkadiusz-jachnik/MLC-PCC/blob/master/mlc_pcc/src/put/mlc/classifiers/f/QuadraticNaiveFMaximizer.java
    """

    def __init__(self, beta):
        self.beta = beta

    def __get_coefficients(self, p):
        """Input: vector p containing probabilities
           Returns the probability that sum(v) = k for all k in {0, length(v)}
           By evaluating a tree structure in a dynamic program
        """
        poly = [[] for i in range(len(p) + 1)]
        poly[0] = [0, 1, 0]

        for i in range(len(p)):
            poly[i + 1] = [0] * (i + 4)
            for j in np.arange(i + 2, 0, -1):
                poly[i + 1][j] = (1 - p[i]) * poly[i][j] + p[i] * poly[i][j - 1]

        return(poly)

    def __algorithm_1(self, p):
        """Return the F_beta scores when the top k instances are predicted as positive, for all k > 1"""

        if self.beta == 1:
            q, r = 1, 1
        elif self.beta == 2:
            q = 2
            r = 1
        n_labels = len(p)

        # line 1: set coefficients C
        poly = self.__get_coefficients(p)

        # line 2: set S
        S = np.zeros(shape=(n_labels * (q + r) + 1), dtype=float)  # We will not use index 0
        for i in np.arange(1, (q + r) * n_labels + 1):
            S[i] = q / (i)

        # line 3: get the f_Beta's
        # Again, index 0 remains unused (fb_0 is not computed, indexing starts at 1)
        fs = [0] * (n_labels + 1)
        for k in np.arange(n_labels, 0, -1):
            for k1 in np.arange(0, k + 1):

                fs[k] += (1 + (r / q)) * k1 * poly[k][k1 + 1] * S[r * k + q * k1]

            # line 5 is done by the above double indexing of poly: go up in the tree for each k
            for i in np.arange(1, (q + r) * (k - 1) + 1):
                # Mind indexing of p here: p is zero-indexed
                S[i] = (1 - p[k - 1]) * S[i] + p[k - 1] * S[i + q]

        return(fs)

    def get_predictions(self, p):
        """Return predictions that maximize the expected F_beta score
        when the probabilities that the labels are positive are given by p"""

        # sort p
        sort_order = np.argsort(p)[::-1]
        p_sorted = p[sort_order]
        fs = self.__algorithm_1(p_sorted)

        maxIndex = np.argmax(fs)

        preds = np.concatenate([np.ones((maxIndex)), np.zeros(
            (len(p) - maxIndex))])[np.argsort(sort_order)].astype(int)
        return(preds)
