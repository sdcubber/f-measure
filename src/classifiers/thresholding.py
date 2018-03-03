# Different thresholding algorithms to optimize F_beta measure
# OptimalMeanThreshold: binary relevance with a single optimized threshold for all instances and all labels.
# The best threshold is computed for all instances in parallell, and the mean is used as the optimal t.
# OptimalGlobalThreshold: binary relevance with a single optimized threshold for all instances and all labels.
# The best threshold is determined by sorting all predicted probalities of the entire dataset ($O(mnlog(mn))$)
# Author: Stijn Decubber

import numpy as np
from tqdm import tqdm


class OptimalMeanThreshold(object):

    def __init__(self, beta):
        self.beta = beta

    def get_optimal_t(self, labels, marginal_predictions):
        H = marginal_predictions
        Y = labels

        H_sort_order = H.argsort(axis=-1)[:, ::-1]
        H_sorted = np.array([H[i][H_sort_order[i]]
                             for i in range(len(H))])  # This should be done in numpy
        Y_sorted = np.array([Y[i][H_sort_order[i]] for i in range(len(H))])

        y = np.sum(Y, axis=1)  # stays fixed
        h = np.zeros(len(H))
        yh = np.zeros(len(H))

        max_F = np.zeros(len(H))
        best_t = np.ones(len(H))

        for i in tqdm(range(H_sorted.shape[1] - 1)):
            h += 1
            yh += Y_sorted[:, i]
            F = (1 + (self.beta**2)) * (yh) / ((self.beta**2) * y + h)
            higher_F = [F > max_F]
            best_t[higher_F] = H_sorted[:, i + 1][higher_F]
            max_F[higher_F] = F[higher_F]

        print('Average max F{}: {}'.format(self.beta, np.mean(max_F)))
        print('Average optimal t: {}'.format(np.mean(best_t)))

        return best_t


class OptimalGlobalThreshold(object):

    def __init__(self, beta):
        self.beta = beta

    def get_optimal_t(self, labels, marginal_predictions):
        # Sort true and predicted row_wise on Hi
        H = marginal_predictions.flatten()
        Y = labels.flatten()

        H_sort_order = np.argsort(H)[::-1]
        H_sorted = H[H_sort_order]
        Y_sorted = Y[H_sort_order]
        instance_order = np.array([[i] * labels.shape[1]
                                   for i in range(labels.shape[0])]).flatten()[H_sort_order]

        y = np.sum(labels, axis=1)  # fixed
        h = np.zeros(len(marginal_predictions))
        yh = np.zeros(len(marginal_predictions))

        F_vector = np.zeros(len(marginal_predictions))

        max_F = 0
        best_t_global = 0

        for i in tqdm(range(len(H_sorted) - 1)):
            row = instance_order[i]
            h[row] += 1
            yh[row] += Y_sorted[i]

            F_vector[row] = (1 + (self.beta**2)) * (yh[row]) / ((self.beta**2) * y[row] + h[row])
            if np.mean(F_vector) > max_F:
                best_t_global = H_sorted[i + 1]
                max_F = np.mean(F_vector)

            if i % 100000 == 0:
                print('Current t: {:.2f} - Best t: {:.2f}'.format(H_sorted[i], best_t_global))

        print('Max F{}: {:.2f} '.format(self.beta, max_F))
        print('Best t: {:.2f}'.format(best_t_global))

        return best_t_global
