"""
Build graph for the use of Graph-based Semi-Supervised Learning.
"""

# Authors: Xuan Zhang <xuanzhang@jhu.edu>
# Lisence: MIT

import numpy as np
import regression
import classification
import logging

class Graph():
    def __init__(self, model, x, y, distance, sparsity, logger, domain_name_lst,
                 distance_param=None, k=5, ind_label=None, num_label=5):
        # model: [regression, LP ...]
        self.model = model
        # x: np.ndarray((n,d)), domain vectors
        self.x = x
        # y: np.ndarray((n,1)), labels
        self.y = y
        # n: number of examples
        self.n = self.x.shape[0]
        self.best_ind = np.argmax(self.y)

        # distance: [euclidean, dotproduct, cosinesim, constant, heuristic]
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
        if self.ind_label is None:
            self._labeled_points()
        if self.model == "LP":
            ind_label_sorted = sorted(self.ind_label, key=lambda k: self.y[k], reverse=True)
            self.y[ind_label_sorted[:int(self.num_label/3)]] = 1
            self.y[ind_label_sorted[int(self.num_label/3):]] = 0

        # domain_name_lst: the name of each domain for a domain vector
        self.domain_name_lst = domain_name_lst

        # weight: the weight matrix
        self.weight = np.zeros((self.n,self.n))

        if self.distance == "euclidean":
            self._euclidean()
        elif self.distance == "dotproduct":
            self._dotproduct()
        elif self.distance == "cosinesim":
            self._cosinesim()
        elif self.distance == "constant":
            self._constant()
        elif self.distance == "heuristic":
            self._heuristic()

        if self.sparsity == "knn":
            self._knn()

        # ind_unlabeled: np.ndarray((u,1)), the indices of unlabeled data points
        self.ind_unlabel = np.setdiff1d(np.arange(self.n), self.ind_label)
        # x_label: np.ndarray((l,d)), labeled domain vectors
        self.x_label = self.x[self.ind_label]
        # y_label: np.ndarray((l,1))
        self.y_label = self.y[self.ind_label]
        # x_unlabel: np.ndarray((u,d)), unlabeled domain vectors
        self.x_unlabel  = self.x[self.ind_unlabel]
        self.num_label = self.x_label.shape[0]

        self.logger = logger

    # def _label2class(self):
    #     '''
    #     Change regression label to class label.
    #     '''
    #     ind_sorted = sorted(range(self.n), key=lambda k: self.y[k],reverse=True)
    #     self.y[ind_sorted[:int(self.n/3)]] = 1
    #     self.y[ind_sorted[int(self.n/3)+1:]] = 0

    def _euclidean(self):
        '''
        Calculate the Euclidean weight.
        '''
        if self.distance_param is None:
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
        if self.distance_param is None:
            self.distance_param = 0.03
        for i in range(self.n):
            for j in range(self.n):
                self.weight[i][j] = np.exp(-1./self.distance_param
                                           * ((1-self.x[i].dot(self.x[j]))
                                              / (np.linalg.norm(self.x[i])*np.linalg.norm(self.x[j]) + 1e-8)))

    def _constant(self):
        '''
        Set all the weights to a constant.
        '''
        if self.distance_param is None:
            self.distance_param = 1
        self.weight = np.ones((self.n, self.n)) * self.distance_param

    def _heuristic(self):
        '''
        Heuristic weight matrix.
        '''
        sim_dict = {'initial_learning_rate':0.1, 'transformer_attention_heads':0.88,
                    'num_layers':0.78, 'transformer_feed_forward_num_hidden':0.9,
                    'num_embed':0.4, 'bpe_symbols':0.55, 'transformer_model_size':0.4 }
        d = self.x.shape[1]
        for i in range(self.n):
            for j in range(self.n):
                for d in range(d):
                    domain = self.domain_name_lst[d]
                    if self.x[i][d] != self.x[j][d]:
                        self.weight[i][j] += sim_dict[domain]
                    else:
                        self.weight[i][j] += 1

    def _knn(self):
        '''
        Get a sparse weight matrix where wi,j=0, if xi is not in xj's k-nearest-neighbourhood.
        '''
        mask = np.full((self.n, self.n), False, bool)
        for i in range(self.n):
            # get the indices of k nearest neighbours of xi
            k_ind = np.argsort(self.weight[i])[-self.k:]
            mask[i][k_ind] = True
            # set elements with indices not in k_ind to 0
            #np.put(self.weight[i], np.setdiff1d(np.arange(self.n), k_ind), 0)
        mask = np.logical_or(mask.T, mask)
        self.weight *= mask

    def _labeled_points(self):
        '''
        Uniformly sample labeled points from all the data.
        '''
        self.ind_label = np.random.choice(self.n, self.num_label)

    def update(self):
        self.logger.info("Labeled points ({0}): ".format(self.num_label))
        self.logger.info(self.x_label)
        self.logger.info(self.y_label)

        if self.model == "regression":
            # 1. train the model
            self.logger.info("Start training ... ")
            reg = regression.Regression(self.x, self.weight, self.ind_label, self.y_label, self.logger)
            reg.train()

            # 2. log info
            best_new_ind = reg.log(self.x, self.y, self.num_label, self.ind_label)
        elif self.model == "LP":
            lp = classification.LP(self.weight, self.ind_label, self.ind_unlabel, self.y_label, self.logger)
            hu = lp.propagate()
            best_new_ind = self.ind_unlabel[np.argmax(hu)]
            lp.log(hu, best_new_ind, self.x, self.y, self.best_ind)

        # 3. update the graph
        self.ind_label = np.append(self.ind_label, best_new_ind)
        self.y_label = self.y[self.ind_label]
        self.x_label = self.x[self.ind_label]
        self.ind_unlabel = np.setdiff1d(np.arange(self.n), self.ind_label)
        self.x_unlabel = self.x[self.ind_unlabel]
        self.num_label = self.num_label + 1





