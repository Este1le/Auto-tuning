import os
import pickle
import numpy as np
from collections import defaultdict
from preprocess import extract_data

def s2best(sampling_order, best_id, y):
    if best_id in sampling_order[:3]:
        return 0
    return sampling_order.index(best_id) - 3 + 1

def s2close_best(sampling_order, best_id, y, closeness=0.5):
    best = y[best_id]
    for i in range(len(sampling_order)):
        s = y[sampling_order[i]]
        if s >= best-closeness:
            if i < 3:
                return 0
            else:
                return i - 3 + 1

def ns2dif_best(sampling_order, best_id, y, n_steps=10):
    best = y[best_id]
    current_best = -1
    for i in sampling_order[:n_steps+3]:
        if y[i] > current_best:
            current_best = y[i]
    return best - current_best

if __name__ == "__main__":
    architectures = ["rnn", "trans"]
    evaluations = [s2best, s2close_best, ns2dif_best]
    input = '/export/a08/xzhan138/Auto-tuning/single_output/'

    for architecture in architectures:
        arch_dir = input + architecture
        if os.path.isdir(arch_dir):
            for dataset_name in os.listdir(arch_dir):
                dataset_dir = arch_dir + "/" + dataset_name
                if os.path.isdir(dataset_dir):
                    evals = defaultdict(list)
                    modeldir = "/export/a10/kduh/p/mt/gridsearch/" + dataset_name + "/models/"
                    for pf in os.listdir(dataset_dir):
                        if "gru" in pf:
                            x,y,_ = extract_data(modeldir, 5, architecture, "gru")
                        else:
                            x,y,_ = extract_data(modeldir, 5, architecture)
                        best_id = np.argmax(y)

                        if pf.endswith("pkl"):
                            with open(pf, 'rb') as f:
                                I = pickle.load(f)
                            for eval in evaluations:
                                I_eval = np.apply_along_axis(eval, 1, I, best_id=best_id, y=y)
                                evals[pf[:-4]].append((np.average(I_eval), np.std(I_eval)))
                    with open(dataset_dir + "/eval.pkl", 'wb') as f:
                        pickle.dump(evals)
