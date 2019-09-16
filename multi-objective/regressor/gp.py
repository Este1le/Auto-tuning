import numpy as np
import sys
sys.path.append('/Users/xuanzhang/miniconda2/lib/python2.7/site-packages/george-0.3.1-py2.7-macosx-10.5-x86_64.egg')
import george
from george import kernels
from base_regressor import BaseRegressor

class GP(BaseRegressor):

    def __init__(self, x, y, init_id):
        super(GP, self).__init__(x, y, init_id)

    def fit_predict(self):
        kernel = kernels.ExpSquaredKernel(np.ones([self.x.shape[1]]), ndim=self.x.shape[1])
        gp = george.GP(kernel)
        gp.compute(self.x[self.init_id])
        pred_mean, pred_var = gp.predict(self.y, self.x, return_var =True)

        return pred_mean, pred_var

