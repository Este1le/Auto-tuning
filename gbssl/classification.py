"""
Graph-based Semi-Supervised Classification.
"""

# Authors: Xuan Zhang <xuanzhang@jhu.edu>
# Lisence: MIT
import numpy as np
import logging

class LP():
    def __init__(self, weight, ind_label, ind_unlabel, y_label, logger):
        self.weight = weight
        self.ind_label = ind_label
        self.ind_unlabel = ind_unlabel
        self.y_label = y_label
        self.logger = logger
        self.n = self.weight.shape[0]
        self.y_new = np.zeros((self.n,1))

    def propagate(self):
        Wuu = self.weight[self.ind_unlabel][self.ind_unlabel]
        Wul = self.weight[self.ind_unlabel][self.ind_label]
        Duu = np.diag(np.sum(Wuu, axis=1))
        Yl = self.y_label
        hu = np.linalg.inv(Duu-Wuu).dot(Wul).dot(Yl)
        self.y_new[self.ind_label] = self.y[self.ind_label]
        self.y_new[self.ind_unlabel] = hu

        return hu

    def log(self, hu,best_new_ind, x, y, best_ind):
        self.logger.info("Next point: ")
        self.logger.info(x[best_new_ind])
        self.logger.info("Estimated label: {0}, Real label: {1}".format(max(hu), y[best_new_ind]))
        self.logger.info("Estimated label for the true best: {0}".format(self.y_new[best_ind]))



