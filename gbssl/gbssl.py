import argparse
import logging
import os
import numpy as np
import preprocess
import graph

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    parser.add_argument('--distance', required=True,
                        choices=['euclidean', 'dotproduct', 'cosinesim', 'constant', 'heuristic'],
                        help='The metric for computing weights on the edges.')
    parser.add_argument('--sparsity', required=True, choices=['full', 'knn'],
                        help="full: create a fully connected graph; "
                             "knn: Nodes i, j are connected by an edge if i is in j's k-nearest-neighbourhood.")
    parser.add_argument('--distance-param', required=False, default=None,
                        help='The parameter for calculating the distance.')
    parser.add_argument('--k', required=False, type=int, default=5,
                        help='The parameter for k-nearest-neighbourhood.')

    # Evaluation arguments
    parser.add_argument('--budget', type=int, default=10,
                        help='This is for the use of measuring the performance given limited budget.\
                        Or the number of models we are allowed to evaluate.')
    parser.add_argument('--dif', type=float, default=1,
                        help='This is for the use of checking how many runs should it take to\
                        to achieve a result whose difference to the best result is no larger than dif.')

    # Running arguments
    parser.add_argument('--num-run', type=int, default=150,
                        help='Number of runs.')

    # Output arguments
    parser.add_argument('--output', required=True, help='The output path.')

    args = parser.parse_args()
    if args.architecture == 'rnn' and 'rnn_cell_type' not in vars(args):
        parser.error('--rnn-cell-type is required if the architecture is rnn.')

    return args

def write_results(args, best_ind, fix_budget, close_best, graph_obj):

    with open(args.output, "w") as f:
        f.write("Labeled points ({0}): \n".format(graph_obj.num_label))
        f.write(str(graph_obj.x_label) + "\n")
        f.write(str(graph_obj.y_label) + "\n\n")

        f.write("########################################\n\n")
        f.write("Best ind\n")
        f.write(str(best_ind) + "\n")
        f.write("Ave: " + str(sum(best_ind)/len(best_ind)) + "\n")
        f.write("Std: " + str(np.std(best_ind)) + "\n\n")

        f.write("Close best({0})\n".format(args.dif))
        f.write(str(close_best) + "\n")
        f.write("Ave: " + str(sum(close_best)/len(close_best)) + "\n")
        f.write("Std: " + str(np.std(close_best)) + "\n\n")

        f.write("Fix budget({0})\n".format(args.budget))
        f.write(str(fix_budget) + "\n")
        f.write("Ave: " + str(sum(fix_budget)/len(fix_budget)) + "\n")
        f.write("Std: " + str(np.std(fix_budget)) + "\n\n")

def main():
    args = get_args()

    # 0. Prepare data
    logger.info("Extracting data ...")
    rescaled_domain_lst, domain_name_lst, eval_lst, BEST, WORST = \
        preprocess.extract_data(args.modeldir, args.architecture, args.rnn_cell_type, args.metric, args.best)
    logger.info("Best point: {0}".format(BEST))
    X = np.array(rescaled_domain_lst)
    Y = np.array(eval_lst)

    best_ind = []
    fix_budget = []
    close_best = []

    for nr in range(args.num_run):
        # 1. setup logger
        logdir = args.output + "_log"
        if not os.path.isdir(logdir):
            os.mkdir(logdir)
        file_handler = logging.FileHandler(logdir+"/"+str(nr)+".log")
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)

        # 2. Initialization
        if nr < X.shape[0]-2:
            start = nr
            end = nr+3
        else:
            start = nr%(X.shape[0]-2)
            end = nr%(X.shape[0]-2)+3
        ind_label = np.arange(start, end)
        num_label = 3

        # 3. Find the best label
        logger.info("Building the graph ...")
        graph_obj = graph.Graph(X, Y, args.distance, args.sparsity, logger, domain_name_lst,
                            args.distance_param, args.k, ind_label, num_label)
        num_update = 1
        while True:
            logger.info("###############################")
            logger.info("Update # {0}: ".format(num_update))
            graph_obj.update()
            if BEST in graph_obj.y_label:
                logger.info("Found the best configuration at update # {0}".format(num_update))
                break
            num_update += 1

        # 4. Write stats
        best_ind.append(graph_obj.y_label.shape[0]-3)
        fix_budget.append(abs(max(graph_obj.y_label[:args.budget])-BEST))
        for i in range(graph_obj.y_label.shape[0]):
            if abs(graph_obj.y_label[i]-BEST) <= args.dif:
                close_best.append(i+1)
                break
        write_results(args, best_ind, fix_budget, close_best, graph_obj)

if __name__ == "__main__":
    main()
