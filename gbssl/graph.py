"""
Build graph for the use of Graph-based Semi-Supervised Learning.
"""

# Authors: Xuan Zhang <xuanzhang@jhu.edu>
# Lisence: MIT

import numpy as np
import regression

class Graph():
    def __init__(self, x, y, distance, sparsity, distance_param=None, k=5, ind_label=None, num_label=5):
        # x: np.ndarray((n,d)), domain vectors
        self.x = x
        # y: np.ndarray((n,1)), labels
        self.y = y
        # distance: [euclidean, dotproduct, cosinesim, constant]
        # metric for computing weights on the edges
        self.distance = distance
        # distance_param: the parameter for calculating the distance
        self.distance_param = distance_param
        # sparsity: [full, knn]
        # full: create a fully connected graph
        # knn: Nodes i, j are connected by an edge if i is in j's k-nearest-neighbourhood
        self.sparsity = sparsity
        # k: parameter for k-nearest-neighbourhood
        self.k = k
        # ind_label: np.ndarray((l,1)), indices of initial labeled points
        self.ind_label = ind_label
        # num_label: int, number of labeled points
        self.num_label = num_label

        # n: number of examples
        self.n = self.x.shape[0]
        # weight: the weight matrix
        self.weight = np.ndarray((n,n))

        if self.distance == "euclidean":
            self._euclidean()
        elif self.distance == "dotproduct":
            self._dotproduct()
        elif self.distance == "cosinesim":
            self._cosinesim()
        elif self.distance == "constant":
            self._constant()

        if self.sparsity == "knn":
            self._knn()

        if self.ind_label == None:
            self._labeled_points()
            # ind_unlabeled: np.ndarray((u,1)), the indices of unlabeled data points
            self.ind_unlabel = np.setdiff1d(np.arange(self.n), self._labeled_points())
        else:
            self.ind_unlabel = np.setdiff1d(np.arange(self.n), self.ind_label)
        # x_label: np.ndarray((l,d)), labeled domain vectors
        self.x_label = self.x[self.ind_label]
        # y_label: np.ndarray((l,1))
        self.y_label = self.y[self.ind_label]
        # x_unlabel: np.ndarray((u,d)), unlabeled domain vectors
        self.x_unlabel  = self.x[self.ind_unlabel]
        self.num_label = self.x_label.shape[0]

    def _euclidean(self):
        '''
        Calculate the Euclidean weight.
        '''
        if self.distance_param == None:
            # set a heuristic param by minimum spanning tree
            # reference: Semi-Supervised Learning with Graphs, Xiaojin Zhu, 2005, section 7.3.
            # x is scaled to [0,1], the maximum difference between xi and xj is d
            self.distance_param = self.x.shape[1]/3.
        for i in range(self.n):
            for j in range(self.n):
                self.weight[i][j] = np.exp(-np.sum(np.square(self.x[i]-self.x[j]))
                                           /np.float(np.square(self.distance_param)))

    def _dotproduct(self):
        '''
        Calculate the dot product weight.
        '''
        for i in range(self.n):
            for j in range(self.n):
                self.weight[i][j] = self.x[i].dot(self.x[j])

    def _cosinesim(self):
        '''
        Calculate the cosine similarity weight.
        '''
        if self.distance_param == None:
            self.distance_param = 0.03
        for i in range(self.n):
            for j in range(self.n):
                self.weight[i][j] = np.exp(-1./self.distance_param
                                           * ((1-self.x[i].dot(self.x[j]))
                                              / (np.linalg.norm(self.x[i])*np.linalg.norm(self.x[j]))))

    def _constant(self):
        '''
        Set all the weights to a constant.
        '''
        if self.distance_param == None:
            self.distance_param = 1
        self.weight = np.ones((self.n, self.n)) * self.distance_param

    def _knn(self):
        '''
        Get a sparse weight matrix where wi,j=0, if xi is not in xj's k-nearest-neighbourhood.
        '''
        for i in range(self.n):
            # get the indices of k nearest neighbours of xi
            k_ind = np.argsort(self.weight[i])[-k:]
            # set elements with indices not in k_ind to 0
            np.put(self.weight[i], np.setdiff1d(np.arange(self.n), k_ind), 0)

    def _labeled_points(self):
        '''
        Uniformly sample labeled points from all the data.
        '''
        self.ind_label = np.random.choice(self.n, self.num_label)

    def update(self):
        reg = regression.Regression(self.x, self.weight, self.ind_label, self.y_label)
        print("Labeled points ({0}): ".format(self.num_label))
        print(self.x_label)
        print(self.y_label)
        print("Start training ... ")
        reg.train()
        y_new = reg.predict(np.pad(self.x), (1,), 'constant', constant_values=(1)).flatten()
        # the best predicted y must be one of the top (num_label+1) predicted labels
        top_new_ind = np.flip(np.argsort(y_new)[-(self.num_label+1):])
        for id in top_new_ind:
            if id in self.ind_label:
                continue
            else:
                best_new_ind = id
                break
        print("Next point: ")
        print(self.x[best_new_ind])
        print("Estimated label: {0}, Real label: {1}".format(y_new, self.y[best_new_ind]))
        self.ind_label = np.append(self.ind_label, best_new_ind)
        self.y_label = self.y[self.ind_label]
        self.x_label = self.x[self.ind_label]
        self.ind_unlabel = np.setdiff1d(np.arange(self.n), self.ind_label)
        self.x_unlabel = self.x[self.ind_unlabel]
        self.num_label = self.num_label + 1





