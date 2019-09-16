import argparse
import pickle
import numpy as np
import os.path
import sys
sys.path.insert(1, '/export/a08/xzhan138/Auto-tuning/multi-objective/regressor')
from gp import GP
from krr import KRR
from gbssl import GBSSL
from pareto import pareto
from preprocess import extract_data

def get_args():
    parser = argparse.ArgumentParser(description="Multi-objective Hyperparameter Optimization.")
    parser.add_argument("--dataset", choices=["ted-zh-en", "ted-ru-en", "robust19-en-ja", "robust19-ja-en"],
                        help="Dataset name.")
    parser.add_argument("--model", choices=["krr", "gp", "gbssl"],
                        help="Optimization algorithm.")
    parser.add_argument("--threshold", type=int, default=5, help="Remove samples with bad performance(BLEU).")
    parser.add_argument("--random-seed", type=int, help="Random seed for random sampling from pareto frontiers.")
    parser.add_argument("--output", help="Output directory.")

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    modeldir = "/export/a10/kduh/p/mt/gridsearch/" + args.dataset + "/models/"
    np.random.seed(args.random_seed)

    x, y1, y2 = extract_data(modeldir=modeldir, threshold=args.threshold)
    y = np.vstack((y1, y2))
    real_ranking = np.array(pareto(y))

    front_file = args.output + "/" + args.dataset + "/fronts.pkl"
    if not os.path.exists(front_file):
        front_ids, = np.where(real_ranking==max(real_ranking))
        with open(front_file,'wb') as fobj:
            pickle.dump(front_ids, fobj)

    result = np.zeros((len(y1)-3, len(y1)))
    for i in range(len(y1)-3):
        print("step {0}/{1}".format(i+1, len(y1)-3))
        label_ids = np.array([i,i+1,i+2])
        while len(label_ids) != len(y1):
            y_preds = np.zeros(y.shape)
            y_vars = [0] * y.shape[1]
            for k in range(y.shape[0]):
                if args.model == "gbssl":
                    opt_model = GBSSL(x, y[k][label_ids], label_ids)
                elif args.model == "gp":
                    opt_model = GP(x, y[k][label_ids], label_ids)
                elif args.model == "krr":
                    opt_model = KRR(x, y[k][label_ids], label_ids)
                y_preds[k], y_vars[k] = opt_model.fit_predict()
                del opt_model

            unlabel_ids = np.array([u for u in range(len(y1)) if u not in label_ids])
            ranking = np.array(pareto(np.array(y_preds)[:,unlabel_ids]))

            current_front_ids = np.array([])
            current_front_ids, = np.where(ranking==max(ranking))
            # while current_front_ids.size == 0:
            #     current_front_ids, = np.where(ranking==max(ranking))
            #     ranking[current_front_ids] = 0
            #     current_front_ids = np.array([c for c in current_front_ids if c not in label_id])

            #current_fronts = y.T[current_front_ids].T
            #current_front_median = np.median(current_fronts, axis=1).reshape(-1,1)
            #current_front_id = current_front_ids[np.argmin(np.abs(np.sum(current_fronts - current_front_median, axis=0)))]
            current_front_id = np.random.choice(current_front_ids)
            label_ids = np.append(label_ids, unlabel_ids[current_front_id])
        result[i] = label_ids

    output_file = args.output + "/" + args.dataset + "/" + args.model + "/" + "random" + str(args.random_seed) + ".pkl"
    with open(output_file,'wb') as fobj:
        pickle.dump(result, fobj)







