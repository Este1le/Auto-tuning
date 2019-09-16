from abc import ABCMeta, abstractmethod

class BaseRegressor(object):
    __metaclass__ = ABCMeta

    def __init__(self, x, y, init_id):
        """
        A base class for regressors.

        :param x: np.ndarray((N, D)).
            N is the number of samples, D is the dimension of each sample.
        :param y: np.ndarray((n, 1)).
            n is the number of labeled data points, equals to len(init_id).
        :param init_id: array_like, shape = [n].
        """
        self.x = x
        self.y = y
        self.init_id = init_id

    @abstractmethod
    def fit_predict(self):
        raise NotImplementedError("The function to fit the data to model and get predictions \
        must be implemented.")




