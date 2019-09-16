import numpy as np
from base_regressor import BaseRegressor

class GBSSL(BaseRegressor):

    def __init__(self, x, y, init_id):
        super(GBSSL, self).__init__(x, y, init_id)
        self.graph = Graph(x, y, init_id)

    def fit_predict(self):
        self.graph.fit()
        pred_lst =  self.graph.F
        del self.graph
        return pred_lst, None

class Graph(object):

    def __init__(self, x, y, init_id, k=20, max_iter=10000):
        self.x = x
        self.y= y
        self.init_id = init_id
        self.k = k
        self.max_iter = max_iter
        self.n = self.x.shape[0]
        self.P = np.ones((self.n, self.n))
        self.B = np.zeros((self.n))
        self.F = np.zeros((self.n))
        self.weight = np.zeros((self.n,self.n))
        self._euclidean()

    def _euclidean(self):
        for i in range(self.n):
            for j in range(self.n):
                self.weight[i][j] = np.exp(-0.5*np.sum(np.square(self.x[i]-self.x[j])))

    def _get_P(self):
        mask = np.full((self.n, self.n), False, bool)
        for i in range(self.n):
            k_ind = np.argsort(self.weight[i])[-self.k:]
            mask[i][k_ind] = True
        mask = np.logical_or(mask.T, mask)
        np.fill_diagonal(mask, False)
        self.weight *= mask

        self.weight = self.weight/self.weight.sum(axis=1)[:,None]

        self.P = np.array(self.weight)
        self.P[self.init_id] = 0

    def _get_B(self):
        self.B[self.init_id] = self.y

    def _propagate(self):
        return self.P.dot(self.F) + self.B

    def fit(self):
        self._get_P()
        self._get_B()
        remaining_iter = self.max_iter
        while remaining_iter > 0:
            self.F = self._propagate()
            remaining_iter -= 1
