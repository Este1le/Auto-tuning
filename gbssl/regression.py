"""
Graph-based Semi-Supervised Regression.
"""

# Authors: Xuan Zhang <xuanzhang@jhu.edu>
# Lisence: MIT
import numpy as np
import logging

class Regression():
    def __init__(self, x, weight, ind_label, y_label, alpha=0.05, tolerance=1e-4, r=0.001):
        # x: np.ndarray((n,d)), domain vectors
        self.x = np.pad(x, ((0,0),(1,0)), 'constant', constant_values=1)
        # weight: np.ndarray((n,n)), the edge weight matrix
        self.weight = weight
        # ind_label: np.ndarray((l,1)), the indices for labeled points
        self.ind_label = ind_label
        # x_label: np.ndarray((l,d+1)), the labeled points
        self.x_label = self.x[self.ind_label]
        # y_label: np.ndarray((l,1)), known labels
        self.y_label = y_label
        self.l = self.y_label.shape[0   ]

        # alpha: learning rate
        self.alpha = alpha
        # tolerance: the model is converged when the difference between new_theta and theta is less than tolerance
        self.tolerance = tolerance
        # r: weight of the second term of the loss
        self.r = r

        self.n = self.x.shape[0]
        self.d = self.x.shape[1]

        # theta: np.ndarray((d+1,1)), initial parameters for the linear function
        self.theta = np.random.rand((self.d))

    def _gradient(self):
        '''
        Get the gradient.
        '''
        first = 0
        y_ = self.predict(self.x_label).tolist()
        y_label = self.y_label.tolist()
        for i in range(self.l):
            first += (y_[i] - y_label[i]) * self.x_label[i]
        first *= 2./self.l

        second = 0
        y_ = self.predict(self.x).tolist()
        weight = self.weight.tolist()
        for a in range(self.n):
            for b in range(self.n):
                second += (y_[a] - y_[b]) * weight[a][b] * (self.x[a]-self.x[b])
        second *= 2*self.r

        gradient = first + second
        return gradient

    def predict(self, x):
        '''
        Get the estimated y.
        :param x: can be a domain matrix or a domain vector.
        :return: the estimated y.
        '''
        return x.dot(self.theta)

    def loss(self):
        '''
        Get the loss/energy.
        '''
        first = 0
        y_ = self.predict(self.x_label).tolist()
        y_label = self.y_label.tolist()
        for i in range(self.l):
            first += (y_[i] - y_label[i])*(y_[i] - y_label[i])
        first *= 1./self.l

        second = 0
        y_ = self.predict(self.x).tolist()
        weight = self.weight.tolist()
        for a in range(self.n):
            for b in range(self.n):
                second += (y_[a] - y_[b]) * (y_[a] - y_[b]) * weight[a][b]
        second *= self.r

        loss = first + second

        return loss

    def train(self):
        '''
        Perform gradient descent.
        '''
        iterations = 1
        old_loss = 0
        while True:
            new_theta = self.theta - self.alpha * self._gradient()

            # stopping condition
            if np.sum(abs(new_theta - self.theta)) < self.tolerance:
                logging.info("Converged.")
                break

            if iterations % 10 == 0:
                new_loss = self.loss()
                logging.info("Iteration: {0} - Loss: {1:.4f}".format(iterations, new_loss))

                if np.abs(old_loss - new_loss) < self.r/10:
                    self.r /= 10

            iterations += 1
            self.theta = new_theta

