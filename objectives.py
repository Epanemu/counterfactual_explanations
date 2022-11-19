"""
Utility function based on wasserstein distance between current and goal distributions.

Example use:

Have some data and classifier
classify the data
for each estimated class, compute distribution over the features of the input

compute set of closest counterfactuals for all the classified points
for the set of counterfactuals and for each class, compute distribution over their features

compute optimal transport for each feature distribution between classes

"""
import ot

def compute_wd(from_data, to_data, is_categorical, bin_edges=None):
    if is_categorical:
        from_unique, from_u_counts = np.unique(from_data, return_counts=True)
        to_unique, to_u_counts = np.unique(to_data, return_counts=True)

        # unify the distributions
        from_counts = []
        to_counts = []
        tot_unique = np.unique(np.concatenate([from_unique, to_unique]))
        # print("UNIQUES")
        # print(tot_unique)
        # print(from_unique)
        # print(to_unique)
        for u_value in tot_unique:
            from_mask = from_unique == u_value
            if np.any(from_mask):
                from_counts.append(from_u_counts[from_mask][0])
            else:
                from_counts.append(0)
            to_mask = to_unique == u_value
            if np.any(to_mask):
                to_counts.append(to_u_counts[to_mask][0])
            else:
                to_counts.append(0)


        from_counts = np.array(from_counts, dtype=np.float64)
        to_counts = np.array(to_counts, dtype=np.float64)

        n = tot_unique.shape[0]
        M = ( np.eye(n) == 0).astype(np.float64) # all distances are equal except to oneself
    else:
        if bin_edges is None:
            from_counts, bin_edges = np.histogram(from_data, bins="fd") # compute the histogram using Freedman Diaconis Estimator for bin width
        else:
            from_counts, bin_edges = np.histogram(from_data, bins=bin_edges)
        to_counts, _ = np.histogram(to_data, bins=bin_edges)
        # print("DATA")
        # print(bin_edges)
        # print(from_data)
        # print(to_data)

        bin_centers = bin_edges[:-1] + (bin_edges[1:] - bin_edges[:-1]) / 2
        M = ot.dist(bin_centers.reshape((-1, 1)), bin_centers.reshape((-1, 1)))
        if M.max() != 0:
            M /= M.max()
        # print()
        # print(M)
        # print(from_counts, to_counts)
        # print(bin_edges)
        # print(from_data, to_data)
    return ot.emd2(from_counts / from_counts.sum(), to_counts / to_counts.sum(), M) # emd2 returns the Earth Mover Distance loss


"""
Utility function based on difference of entropies between current and goal distributions.

Example use:

Have some data and classifier
classify the data
for each estimated class, compute distribution over the features of the input

compute set of closest counterfactuals for all the classified points
for the set of counterfactuals and for each class, compute distribution over their features

compute entropy of all distributions
compute the difference between counterfactual distribution and classified distribution for each class

(or compute the entropy of a distribution on some difference of (average or median or sth) datapoints.)
(or compute entropy of distance of counterfactuals to the original datapoint (weighted by the distance to optimum?), some kind of robustness measure?)

"""

from scipy.stats import entropy
import numpy as np

def compute_entropy(data, is_categorical=False, bin_edges=None):
    """
    Takes one dimensional numpy array of size N
        and bool is_categorical that is True if the data is categorical
    If parameter bin_edges is set, this will be the the bins used, otherwise a new estimate will be made.

    returns the entropy of the corresponding (histogram estimated) distribution and the bin edges used (None if unused)
    """
    if is_categorical:
        _, counts = np.unique(data, return_counts=True)
    else:
        if bin_edges is None:
            counts, bin_edges = np.histogram(data, bins="fd") # compute the histogram using Freedman Diaconis Estimator for bin width
        else:
            counts, bin_edges = np.histogram(data, bins=bin_edges)


    return entropy(counts), bin_edges # entropy function normalizes the data to sum to 1


