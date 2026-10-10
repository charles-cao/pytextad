# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

from numpy import percentile

def adjusted_sigmoid(x, a=1):
    return  1 / (1 + np.exp(-a * (x - 0.5)))
import time
def sigmoid(x):
  # if  x < -1e-7:
  #    return 0
  tmp =  1 / (1 +np.exp(-x))
  return tmp

def abfall(x):
    return np.where((x > 0.75) & (x <= 1), 1, x)


import numpy as np
from sklearn.metrics import euclidean_distances
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted, check_random_state
from sklearn.utils import column_or_1d

import math

MIN_FLOAT = np.finfo(float).eps
MAX_INT = np.iinfo(np.int32).max
MIN_INT = -1 * MAX_INT

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 180

def reoder_score(scores, method='multiplication'):

    scores = column_or_1d(scores)

    if method == 'multiplication':
        return scores.ravel() * -1

    if method == 'subtraction':
        return (scores.max() - scores).ravel()



class ADERH():


    def __init__(self,
                 n_estimators=256,
                 n=18,
                 contamination=0.1,
                 random_state=None,
                 data_name=None,
                 index = None):
        self.n_estimators = n_estimators
        self.nn = n
        self.random_state = random_state
        self.contamination = contamination
        self.data_name = data_name
        self.index = index

    def predit(self):
        self.threshold_ = percentile(self.outlier_score,
                                     100 * (1 - self.contamination))
        self.labels_ = (self.outlier_score > self.threshold_).astype(
            'int').ravel()


    def fit(self, X, y=None):


        # Check data
        X = check_array(X, accept_sparse=False)


        self._fit(X)
        self.outlier_score = reoder_score(self._score_samples(X))
        self.predit()
        return self
    def _normalized_vector(self, X, Y):

        diff = Y - X

        norm = np.linalg.norm(diff)

        if norm == 0:
            return np.zeros_like(diff)

        return diff / norm, norm

    def  _nCircle(self, X, cnnIndex, n, distMatrix, center_index):

        X = np.append(X, X[cnnIndex], axis=0)
        radius_matrix = np.repeat(distMatrix.reshape(-1,1)/2, 2, axis=1).reshape(-1, 1)

        return X, radius_matrix.reshape(-1)



    def _loop_translate(self, a, d):
            n = np.ndarray(a.shape)
            for k in d:
                n[a == k] = d[k]
            return n
    def _fit(self, X):

        n_samples, n_features = X.shape
        self.max_samples_ = self.nn
        self._center = np.empty(
            [self.n_estimators, self.nn, n_features])
        self._ratio = np.empty([self.n_estimators, self.nn])
        self._centroids_radius = np.empty(
            [self.n_estimators, self.max_samples_])
        factor = 2
        factor2  = 2
        self._center2 = np.empty(
            [self.n_estimators, factor2 * self.max_samples_, n_features])
        self._centroids_radius2 = np.empty(
            [self.n_estimators, factor2 * self.max_samples_])
        random_state = check_random_state(self.random_state)
        self._seeds = random_state.randint(MAX_INT, size=self.n_estimators)

        self.colum_list = []

        for i in range(self.n_estimators):
            rnd = check_random_state(self._seeds[i])

            center_index = rnd.choice(
                n_samples, self.max_samples_, replace=False)

            self._center[i] = X[center_index]
            radom_neigh_idx = rnd.choice(
                self.max_samples_, self.max_samples_, replace=False)
            random_neigh_idx = radom_neigh_idx

            for ii in range(len(radom_neigh_idx)):
                if random_neigh_idx[ii] == ii:
                    swap_idx = (ii + 1) % len(radom_neigh_idx)
                    random_neigh_idx[ii], random_neigh_idx[swap_idx] = random_neigh_idx[swap_idx], random_neigh_idx[ii]
            radom_neigh_idx = random_neigh_idx
            random_neigh_dist =  np.sum ((((self._center[i] - X[radom_neigh_idx]) ** 2)), axis=1)

            self._centroids_radius[i] = random_neigh_dist

            cnn_index = radom_neigh_idx
            self._center2[i], self._centroids_radius2[i] =  self._nCircle(self._center[i], cnn_index, factor, random_neigh_dist, center_index)


        return self

    def decision_function(self, X):
        return reoder_score(self._score_samples(X))

    def _eudis5(self,v1, v2):
        dist = [(a - b) ** 2 for a, b in zip(v1, v2)]
        dist = math.sqrt(sum(dist))
        return dist
    def _score_samples(self, X):

        X = check_array(X, accept_sparse=False)
        aderh_score = np.ones([self.n_estimators, X.shape[0]])

        self._center, self._centroids_radius = self._center2, self._centroids_radius2

        for i in range(self.n_estimators):
            x_dists = euclidean_distances(X, self._center[i], squared=True)
            self._centroids_radius[i] = np.where(
                self._centroids_radius[i] !=0,
                self._centroids_radius[i],1)

            cover_radius = np.where(
               x_dists <=  self._centroids_radius[i],
              x_dists, np.nan)
            x_covered = np.where(~np.isnan(cover_radius).all(axis=1))

            cnn_x = np.nanargmin(cover_radius[x_covered], axis=1)
            unique, counts = np.unique(cnn_x, return_counts=True)
            dicc = dict(zip(unique, counts))

            v = cover_radius[x_covered[0], cnn_x]/self._centroids_radius2[i][cnn_x]


            den = self._loop_translate(cnn_x, dicc)
            den /=self._centroids_radius2[i][cnn_x]
            if den.size == 0:
                continue
            den = (den/den.max())
            aderh_score[i][x_covered] = (1-den) * v


        outlier_score = np.mean(aderh_score, axis=0)

        return -outlier_score