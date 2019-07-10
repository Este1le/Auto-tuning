import argparse
import numpy as np
import preprocess
import graph

def get_args():
    parser = argparse.ArgumentParser(description='Graph-Based Semi-Supervised Learning.')

    # Model and Architecture arguments
    parser.add_argument('--modeldir', '-d', required=True, default='/export/a10/kduh/p/mt/gridsearch/ted-zh-en/models/',
                        help='The directory to all the models.')
    parser.add_argument('--architecture', '-a', required=True, choices=['rnn', 'cnn', 'trans'],
                        help='The architecture of the models to be tuned.')
    parser.add_argument('--rnn-cell-type', '-c', required=False, choices=['lstm', 'gru'],
                        help='The cell type of rnn model, required only when the architecture is rnn.')

    # Objective Metric arguments
    parser.add_argument('--metric', required=True, choices=['dev_ppl', 'dev_bleu'],
                        help='The objective metric.')
    parser.add_argument('--best', type=str, required=True, choices=['min', 'max'],
                        help='How should we get the best evaluation result in the pool? By min or max?')

    # Graph arguments
    parser.add_argument('--distance', required=True, choices=['euclidean', 'dotproduct', 'cosinesim', 'constant'],
                        help='The metric for computing weights on the edges.')
    parser.add_argument('--sparsity', required=True, choices=['full', 'knn'],
                        help="full: create a fully connected graph; "
                             "knn: Nodes i, j are connected by an edge if i is in j's k-nearest-neighbourhood.")
    parser.add_argument('--distance-param', required=False, default=None,
                        help='The parameter for calculating the distance.')
    parser.add_argument('--k', required=False, default=5,
                        help='The parameter for k-nearest-neighbourhood.')

    args = parser.parse_args()
    if args.architecture == 'rnn' and 'rnn_cell_type' not in vars(args):
        parser.error('--rnn-cell-type is required if the architecture is rnn.')

    return args

def main():
    args = get_args()

    print("Extracting data ...")
    rescaled_domain_lst, domain_name_lst, eval_lst, BEST, WORST = \
        preprocess.extract_data(args.modeldir, args.architecture, args.rnn_cell_type, args.metric, args.best)

    X = np.array(rescaled_domain_lst)
    Y = np.array(eval_lst)

    ind_label = np.arange(3)
    num_label = 3

    print("Building the graph ...")
    graph_obj = graph.Graph(X, Y, args.distance, args.sparsity,
                        args.distance_param, args.k, ind_label, num_label)
    num_update = 1
    while True:
        print("###############################")
        print("Update # {0}: ".format(num_update))
        graph_obj.update()
        if BEST in graph_obj.y_label:
            print("Found the best configuration at update # {0}".format(num_update))
            break
        num_update += 1

if __name__ == "__main__":
    main()
