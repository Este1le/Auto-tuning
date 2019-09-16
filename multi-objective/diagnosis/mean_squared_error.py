import sys
sys.path.insert(1, '/export/a08/xzhan138/Auto-tuning/multi-objective/regressor')
sys.path.insert(1, '/export/a08/xzhan138/Auto-tuning/multi-objective')
import pickle
import numpy as np
import random
from gp import GP
from krr import KRR
from gbssl import GBSSL
from preprocess import extract_data

if __name__ == "__main__":
    modeldir = "/export/a10/kduh/p/mt/gridsearch/robust19-ja-en/models/"
    x, y, _ = extract_data(modeldir, 5)
    output = "/export/a08/xzhan138/Auto-tuning/diagnosis_output/mse{0}.pkl"
    models = [KRR, GP, GBSSL]
    for i in random.sample(range(len(y)-3), 5):
        print("init: {0}".format(i))
        mse_dic = {}
        for m in range(len(models)):
            mse = []
            label_ids = np.array([i, i+1, i+2])
            model = models[m]
            print("model: {0}".format(model.__name__))
            while len(label_ids) != len(y):
                opt_model = model(x, y[label_ids], label_ids)
                y_preds, y_vars = opt_model.fit_predict()
                del opt_model

                unlabel_ids = np.array([u for u in range(len(y)) if u not in label_ids])
                next_label_id = unlabel_ids[np.argmax(y_preds[unlabel_ids])]
                label_ids = np.append(label_ids, next_label_id)
                mse.append(((y_preds - y)**2).mean())
            mse_dic[model.__name__] = (mse, label_ids)

        with open(output.format(i), 'wb') as f:
            pickle.dump(mse_dic, f, protocol=2)











