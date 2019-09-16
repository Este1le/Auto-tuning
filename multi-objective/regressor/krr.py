from sklearn.kernel_ridge import KernelRidge
from base_regressor import BaseRegressor

class KRR(BaseRegressor):

    def __init__(self, x, y, init_id):
        super(KRR, self).__init__(x, y, init_id)

    def fit_predict(self):
        krr = KernelRidge(kernel='rbf', gamma=0.5)
        krr.fit(self.x[self.init_id], self.y)
        return krr.predict(self.x), None
