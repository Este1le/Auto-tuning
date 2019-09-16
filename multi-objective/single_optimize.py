import argparse
import pickle
import numpy as np
from multiprocessing import Pool
import os.path
import sys
sys.path.insert(1, '/export/a08/xzhan138/Auto-tuning/multi-objective/regressor')
from gp import GP
from krr import KRR
from gbssl import GBSSL
from preprocess import extract_data

def get_args():
    parser = argparse.ArgumentParser(description="Multi-objective Hyperparameter Optimization.")
    parser.add_argument("--dataset", choices=["ted-zh-en", "ted-ru-en", "robust19-en-ja", "robust19-ja-en"])
    parser.add_argument("--architecture", choices=["rnn", "trans"])
    parser.add_argument("--rnn-cell-type", choices=["lstm", "gru"])
    parser.add_argument("--model", choices=["krr", "gp", "gbssl"], help="Optimization algorithm.")
    parser.add_argument("--acquisition", default="max", choices=["max", "erm"], help="Acquisition function.")
    parser.add_argument("--threshold", type=int, default=5, help="Remove samples with bad performance(BLEU).")
    parser.add_argument("--output", help="Output directory.")

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    modeldir = "/export/a10/kduh/p/mt/gridsearch/" + args.dataset + "/models/"

    x, y, _ = extract_data(modeldir=modeldir, threshold=args.threshold,
                           architecture=args.architecture, rnn_cell_type=args.rnn_cell_type)

    result = np.zeros((len(y)-3, len(y)))
    for i in range(len(y)-3):
        print("step {0}/{1}".format(i+1, len(y)-3))
        label_ids = np.array([i,i+1,i+2])
        while len(label_ids) != len(y):
            if args.model == "gbssl":
                opt_model = GBSSL(x, y[label_ids], label_ids)
            elif args.model == "gp":
                opt_model = GP(x, y[label_ids], label_ids)
            elif args.model == "krr":
                opt_model = KRR(x, y[label_ids], label_ids)
            y_preds, y_vars = opt_model.fit_predict()
            del opt_model
            unlabel_ids = np.array([u for u in range(len(y)) if u not in label_ids])

            def get_risk(candidate_id):
                opt_model = GBSSL(x, np.append(y[label_ids], y_preds[candidate_id]), np.append(label_ids, candidate_id))
                new_y_preds, new_y_vars = opt_model.fit_predict()
                del opt_model
                return np.linalg.norm(np.array(new_y_preds)[label_ids] - y[label_ids])

            if args.acquisition == "max":
                next_label_id = unlabel_ids[np.argmax(y_preds[unlabel_ids])]
            elif (args.model == "gbssl") and (args.acquisition == "erm"):
                next_unlabel_ids = np.argsort(y_preds[unlabel_ids])[::-1][:10]
                candidate_ids = unlabel_ids[next_unlabel_ids]
                p = Pool(10)
                risks = p.map(get_risk, candidate_ids)
                next_label_id = candidate_ids[np.argmin(risks)]
                p.close()
                p.join()

            label_ids = np.append(label_ids, next_label_id)
        print(label_ids)
        result[i] = label_ids

    model_name = args.model
    if args.architecture == "rnn":
        model_name += "_{0}".format(args.rnn_cell_type)
    output_file = args.output + "/" + args.architecture + "/" + args.dataset + "/" + \
                  model_name  + "_" + args.acquisition + ".pkl"
    with open(output_file,'wb') as fobj:
        pickle.dump(result, fobj)
