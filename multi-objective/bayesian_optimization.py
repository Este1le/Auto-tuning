import argparse
import numpy as np
import pickle
from robo.fmin import bayesian_optimization
from preprocess import extract_data
import get_kernel

def get_args():
    parser = argparse.ArgumentParser(description='Bayesian Optimization.')

    # Model and Architecture arguments
    parser.add_argument('--dataset', '-d', required=True, default="robust19-ja-en",
                        choices=["robust19-ja-en", "robust19-en-ja", "ted-zh-en", "ted-ru-en"],
                        help='The directory to all the models.')
    parser.add_argument('--architecture', '-a', required=True, default="trans", choices=['rnn', 'cnn', 'trans'],
                        help='The architecture of the models to be tuned.')
    parser.add_argument('--rnn-cell-type', '-c', required=False, choices=['lstm', 'gru'],
                        help='The cell type of rnn model, required only when the architecture is rnn.')

    # Bayesian Optimization arguments
    parser.add_argument('--sampling-method', default="exact", choices=["approx", "exact"],
                        help='Specify the method to choose next sample to update model.\
                        approx: choose the sample in the candidate pool that is closest (measured by distance\
                        arg) to the one returned from maximizing acquisition function.\
                        exact: evaluate all samples in the candidate pool on acquisition function\
                        and choose the one with maximum output.')
    parser.add_argument('--acquisition-func', required=True, choices=["ei", "log_ei", "lcb", "pi"],
                        help="The acquisition function.")
    parser.add_argument('--model-type', default="gp", choices=["gp", "gp_mcmc", "rf", "bohamiann", "dngo"],
                        help="The model type.")
    parser.add_argument('--kernel', default="expsquared", type=str, choices=["constant", "polynomial", "linear",
                        "dotproduct", "exp", "expsquared", "matern32", "matern52",
                        "rationalquadratic", "expsine2", "heuristic"],
                        help='Specify the kernel for Gaussian process.')

    # Other arguments
    parser.add_argument("--threshold", type=int, default=5, help="Remove samples with bad performance(BLEU).")
    parser.add_argument("--output", help="Output directory.")

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    modeldir = "/export/a10/kduh/p/mt/gridsearch/" + args.dataset + "/models/"

    x, y, _ = extract_data(modeldir=modeldir, threshold=args.threshold,
                           architecture=args.architecture, rnn_cell_type=args.rnn_cell_type)
    y = -y

    lower = np.zeros((x.shape[1]))
    upper = np.ones((x.shape[1]))
    if args.architecture == "trans":
        domain_name_lst = ['num_layers', 'transformer_attention_heads', 'transformer_feed_forward_num_hidden',
        'transformer_model_size', 'bpe_symbols', 'initial_learning_rate', 'num_embed']
    else:
        domain_name_lst = []
    kernel = get_kernel.get_kernel(args.architecture, args.kernel, domain_name_lst, x.shape[1])

    def objective_function(sample):
        '''
        Map a sampled domain to evaluation results returned from the model
        :param x_sample: domain sampled from bayesian optimization
        :return: the corresponding evaluation result
        '''
        for i in range(x.shape[0]):
            if (sample == x[i]).all():
                return y[i]

    result = np.zeros((len(y)-3, len(y)))
    for i in range(len(y)-3):
        print("step {0}/{1}".format(i+1, len(y)-3))
        label_ids = np.array([i, i+1, i+2])
        pool = np.array([x[i] for i in range(x.shape[0]) if i not in label_ids])

        bo_res = bayesian_optimization(objective_function, lower, upper, acquisition_func=args.acquisition_func,
                                       model_type=args.model_type, num_iterations=len(y),
                                       X_init=x[label_ids], Y_init=y[label_ids], kernel=kernel,
                                       sampling_method=args.sampling_method, replacement=False,
                                       pool=pool, best=100000)
        x_samples = np.array(bo_res['X'])
        print(x_samples.shape[0])
        sampling_order = []
        for j in range(x_samples.shape[0]):
            for k in range(x.shape[0]):
                if (x_samples[j] == x[k]).all():
                    sampling_order.append(k)
        print(sampling_order)

        result[i] = np.array(sampling_order)

    model_name = "bo_" + args.acquisition_func + "_" + args.kernel + "_" + args.model_type
    output_file = args.output + "/" + args.architecture + "/" + args.dataset + "/" + \
                  model_name + ".pkl"
    with open(output_file, 'wb') as fobj:
        pickle.dump(result, fobj)



