"""
Based at https://github.com/ZigaSajovic/Consensus_Clustering
"""

import bisect
from itertools import combinations

import numpy as np
from sklearn.cluster import k_means


class ConsensusClusterer:
    """
      Implementation of Consensus clustering, following the paper
      https://link.springer.com/content/pdf/10.1023%2FA%3A1023949509487.pdf
    """

    def __init__(self, cluster=k_means, ):
        self.cluster_ = cluster
        self.min_clust_num = None
        self.max_clust_num = None
        self.num_inputs = None
        self.matrix = None
        self.cdf_area = None
        self.cdf_area_diff = None
        self.best_clust_num = None

    def fit(self, predicted_clusters_data, verbose=False):
        predicted_clusters = predicted_clusters_data.features
        self.num_inputs = predicted_clusters.shape[1]

        clust_nums = []
        for col_id in range(self.num_inputs):
            prediction = predicted_clusters[:, col_id]
            num_clust = len(set(prediction))
            if self.min_clust_num is None or self.min_clust_num > num_clust:
                self.min_clust_num = num_clust
            if self.max_clust_num is None or self.max_clust_num < num_clust:
                self.max_clust_num = num_clust
            clust_nums.append(num_clust)

        matrix = np.zeros((len(clust_nums),
                           predicted_clusters.shape[0], predicted_clusters.shape[0]))

        matrix_is = np.zeros((predicted_clusters.shape[0],) * 2)
        for i_ in range(len(clust_nums)):
            # i_ = k - self.min_clust_num
            # for _ in range(self.num_inputs):

            indices = list(range(predicted_clusters.shape[0]))
            # find indexes of elements from same clusters with bisection
            # on sorted array => this is more efficient than brute force search
            id_clusts = np.argsort(predicted_clusters[:, i_])
            sorted_ = predicted_clusters[:, i_][id_clusts]
            for i in range(clust_nums[i_]):  # for each cluster
                ia = bisect.bisect_left(sorted_, i)
                ib = bisect.bisect_right(sorted_, i)
                is_ = id_clusts[ia:ib]
                ids_ = np.array(list(combinations(is_, 2))).T
                # sometimes only one element is in a cluster (no combinations)
                if ids_.size != 0:
                    matrix[i_, ids_[0], ids_[1]] += 1
            # increment counts
            ids_2 = np.array(list(combinations(indices, 2))).T
            matrix_is[ids_2[0], ids_2[1]] += 1

            matrix[i_] /= matrix_is + 1e-8  # consensus matrix
            # Mk[i_] is upper triangular (with zeros on diagonal), we now make it symmetric
            matrix[i_] += matrix[i_].T
            matrix[i_, range(predicted_clusters.shape[0]),
                   range(predicted_clusters.shape[0])] = 1  # always with self
            matrix_is.fill(0)  # reset counter
        self.matrix = matrix
        # fits areas under the CDFs
        self.cdf_area = np.zeros(len(clust_nums))
        for i, m in enumerate(matrix):
            hist, bins = np.histogram(m.ravel(), density=True)
            self.cdf_area[i] = np.sum(h * (b - a)
                                      for b, a, h in zip(bins[1:], bins[:-1], np.cumsum(hist)))
        # fits differences between areas under CDFs
        self.cdf_area_diff = np.array([(Ab - Aa) / Aa if i > 2 else Aa
                                       for Ab, Aa, i in zip(self.cdf_area[1:], self.cdf_area[:-1],
                                                            range(self.min_clust_num, self.max_clust_num - 1))])
        self.best_clust_num = np.argmax(self.cdf_area_diff) + \
                              self.min_clust_num if self.cdf_area_diff.size > 0 else self.min_clust_num

    def predict(self, data):
        """
        Predicts on the data, for best found cluster number
        Args:
          * data -> (examples,attributes) format
        """
        assert self.matrix is not None, "First run fit"
        return k_means(data.features, n_clusters=self.best_clust_num)[1]
