import pickle
import numpy as np

def s2one(sampling_order, fronts):
    step = -3
    for i in range(len(sampling_order)):
        step += 1
        s = sampling_order[i]
        if s in fronts:
            break
    if step < 0:
        step = 0
    return step

def s2all(sampling_order, fronts):
    step = -3
    for i in range(len(sampling_order)):
        step += 1
        s = sampling_order[i]
        if s in fronts:
            fronts = fronts[fronts!=s]
            if len(fronts) == 0:
                break
    if step < 0:
        step = 0
    return step

def fs2n(sampling_order, fronts, n_step=10):
    n_front = 0
    for i in range(n_step+3):
        s = sampling_order[i]
        if s in fronts:
            n_front += 1
    return n_front

if __name__ == "__main__":
    datasets = ["ted-zh-en", "ted-ru-en", "robust19-en-ja", "robust19-ja-en"]
    evals = [s2one, s2all, fs2n]
    models = ["krr", "gp", "gbssl", "random"]
    random_seeds = [59, 18, 20, 73, 61, 29, 58, 65, 14, 30]
    input = '/export/a08/xzhan138/Auto-tuning/multi_output/'

    for dataset in datasets:
        with open(input+dataset+"/fronts.pkl", 'rb') as f:
            fronts = pickle.load(f)
        M = [] # M*R*I*O
        for model in models:
            R = [] # R*I*O
            for r in random_seeds:
                with open(input + dataset + "/" + model + "/random" + str(r) + ".pkl", 'rb') as f:
                    I = pickle.load(f)
                R.append(I)
            R = np.array(R)
            M.append(R)
        M = np.array(M)

        for eval in evals:
            output = "/export/a08/xzhan138/Auto-tuning/multi_output/" + dataset + "/eval_" + eval.__name__ + "_{0}.pkl"
            table1 = [] # n_model * n_random
            for m in M:
                r_lst = [] # n_random
                for r in m:
                    r_eval = np.apply_along_axis(eval, 1, r, fronts=fronts)
                    r_lst.append((np.average(r_eval), np.std(r_eval)))
                table1.append(r_lst)
            with open(output.format("model_random"), 'wb') as f:
                pickle.dump(table1, f)

            table2 = []
            for m in M:
                i_lst = [] # n_init
                for i in np.transpose(m, (1,0,2)): #I*R*O
                    i_eval = np.apply_along_axis(eval, 1, i, fronts=fronts)
                    i_lst.append((np.average(i_eval), np.std(i_eval)))
                table2.append(i_lst)
            with open(output.format("model_init"), 'wb') as f:
                pickle.dump(table2, f)









