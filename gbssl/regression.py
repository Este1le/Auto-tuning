"""
Graph-based Semi-Supervised Regression.
"""

# Authors: Xuan Zhang <xuanzhang@jhu.edu>
# Lisence: MIT
import numpy as np

class Regression():
    def __init__(self, x, weight, ind_label, y_label, alpha=0.5, tolerance=1e-5, r=1):
        # x: np.ndarray((n,d)), domain vectors
        self.x = np.pad(x, (1,), 'constant', constant_values=(1))
        # weight: np.ndarray((n,n)), the edge weight matrix
        self.weight = weight
        # ind_label: np.ndarray((l,1)), the indices for labeled points
        self.ind_label = ind_label
        # x_label: np.ndarray((l,d+1)), the labeled points
        self.x_label = self.x[self.ind_label]
        # y_label: np.ndarray((l,1)), known labels
        self.y_label = y_label

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
        self.gradient = 0
        self.loss = 0

    def _gradient(self):
        '''
        Get the gradient.
        '''
        first = 0
        for i in range(self.l):
            y_ = self.predict(self.x_label[i])
            y = self.y_label[i]
            first += (y_ - y) * self.x_label[i]
        first *= 2./self.l

        second = 0
        for a in range(self.n):
            for b in range(self.n):
                ya_ = self.predict(self.x[a])
                yb_ = self.predict(self.x[b])
                second += (ya_ - yb_) * self.weight[a][b] * (self.x[a]-self.x[b])
        second *= 2*self.r

        self._gradient = first + second

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
        for i in range(self.l):
            y_ = self.predict(self.x_label[i])
            y = self.y_label[i]
            first += np.square(y_ - y)
        first *= 1./self.l

        second = 0
        for a in range(self.n):
            for b in range(self.n):
                ya_ = self.predict(self.x[a])
                yb_ = self.predict(self.x[b])
                second += np.square(ya_ - yb_) * self.weight[a][b]
        second *= r

        self.loss = first + second

    def train(self):
        '''
        Perform gradient descent.
        '''
        iterations = 1
        while True:
            new_theta = self.theta - self.alpha * self._gradient()

            # stopping condition
            if np.sum(abs(new_theta - self.theta)) < self.tolerance:
                print("Converged.")
                break

            if iterations % 100 == 0:
                print("Iteration: {0} - Loss: {1:.4f}".format(iterations, self.loss()))

            iterations += 1
            self.theta = new_theta

