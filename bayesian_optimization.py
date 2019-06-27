import argparse
import logging
import numpy as np
from sklearn import metrics
from robo.fmin import bayesian_optimization
import preprocess
import rescale
import get_kernel

logging.basicConfig(level=logging.INFO)

def get_args():
    parser = argparse.ArgumentParser(description='Bayesian optimization.')

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

    # Bayesian Optimization arguments 
    parser.add_argument('--sampling-method', required=True, choices=["origin", "approx", "exact"],
                        help='Specify the method to choose next sample to update model.\
                        approx: choose the sample in the candidate pool that is closest (measured by distance\
                        arg) to the one returned from maximizing acquisition function.\
                        exact: evaluate all samples in the candidate pool on acquisition function\
                        and choose the one with maximum output.')
    parser.add_argument('--replacement', action='store_true',
                        help='Whether to sample from pool with replacement.')
    parser.add_argument('--acquisition-func', required=True, choices=["ei", "log_ei", "lcb", "pi"],
                        help="The acquisition function.")
    parser.add_argument('--model-type', required=True, choices=["gp", "gp_mcmc", "rf", "bohamiann", "dngo"],
                        help="The model type.")
    parser.add_argument('--num-run', type=int, default=100,
                        help='Number of BO runs.')

    # Gaussian Process arguments
    parser.add_argument('--kernel', required=True, type=str, choices=["constant", "polynomial", "linear", "dotproduct",
                        "exp", "expsquared", "matern32", "matern52", "rationalquadratic","cosine", "expsine2", "heuristic"],
                        help='Specify the kernel for Gaussian process.')

    # Evaluation arguments
    parser.add_argument('--budget', type=int, default=10,
                        help='This is for the use of measuring the performance of BO given limited budget.\
                        Or the number of models we are allowed to evaluate. \
                        Or the number of iterations of BO we are allowed to run.')
    parser.add_argument('--dif', type=float, default=1,
                        help='This is for the use of checking how many iterations should BO take to\
                        to achieve a result whose difference to the best result is no larger than dif.')

    # Output arguments
    parser.add_argument('--output', required=True, help='The output path.')

    args = parser.parse_args()

    if args.architecture=='rnn' and 'rnn_cell_type' not in vars(args):
        parser.error('--rnn-cell-type is required if the architecture is rnn.')

    return args

def extract_data(args):
    logging.info("Extracting domains and evaluation results for all models")
    domain_eval_lst = preprocess.get_all_domain_eval(args.modeldir)
    if args.architecture == 'rnn':
        domain_eval_lst_arch = [de for de in domain_eval_lst if (de[0]['architecture']==args.architecture) and (de[0]['rnn_cell_type']==args.rnn_cell_type)]
    else:
        domain_eval_lst_arch = [de for de in domain_eval_lst if (de[0]['architecture']==args.architecture)]

    domain_dict_lst, eval_dict_lst = list(list(zip(*domain_eval_lst_arch))[0]), list(list(zip(*domain_eval_lst_arch))[1])

    # Rescale values to the range [0,1] and turn dictionary into list
    if args.architecture=='rnn':
        rescaled_domain_lst, domain_name_lst = rescale.rescale(domain_dict_lst, rescale.rnn_rescale_dict)
    elif args.architecture=='cnn':
        rescaled_domain_lst, domain_name_lst = rescale.rescale(domain_dict_lst, rescale.cnn_rescale_dict)
    elif args.architecture=='trans':
        rescaled_domain_lst, domain_name_lst = rescale.rescale(domain_dict_lst, rescale.trans_rescale_dict)

    # The objective we want to optimize
    if args.best == 'min':
        eval_lst = [e[args.metric] for e in eval_dict_lst]
        WORST = 100000
    else:
        eval_lst = [-e[args.metric] for e in eval_dict_lst]
        WORST = 0
    BEST = min(eval_lst)

    return rescaled_domain_lst, domain_name_lst, eval_lst, BEST, WORST

def write_results(args, results, best_ind, fix_budget, close_best):

    with open(args.output, "w") as f:
        f.write(str(results))
        f.write("\n")
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

def run_bayesian_optimization(args, kernel, objective_function, rescaled_domain_lst, eval_lst, BEST, WORST):
    origin_rdl = rescaled_domain_lst[:]
    origin_el = eval_lst[:]

    lower = np.array([0]*len(rescaled_domain_lst[0]))
    upper = np.array([1]*len(rescaled_domain_lst[0]))

    results = []
    best_ind = []
    fix_budget = []
    close_best = []

    for _ in range(args.num_run):
        rescaled_domain_lst = origin_rdl[:]
        eval_lst = origin_el[:]
        logging.info("#" + str(_) + " run of bayesian optimization.")
        result = bayesian_optimization(objective_function, lower,upper, acquisition_func=args.acquisition_func, 
                                       model_type=args.model_type, num_iterations=len(origin_rdl), kernel=kernel, 
                                       sampling_method=args.sampling_method, replacement=args.replacement, 
                                       pool=np.array(rescaled_domain_lst), best=BEST)
        results.append(result)
        invs = result["incumbent_values"]

        if BEST in invs:
            best_ind.append(invs.index(BEST)+1)
        else:
            best_ind.append(WORST)

        fix_budget.append(abs(min(invs[:args.budget])-BEST))

        for i in range(len(invs)):
            if invs[i]-BEST <= args.dif:
                close_best.append(i+1)
                break

    return results, best_ind, fix_budget, close_best

def main():
    args = get_args()
    rescaled_domain_lst, domain_name_lst, eval_lst, BEST, WORST = extract_data(args)
    kernel = get_kernel(args.architecture, rescaled_domain_lst, domain_name_lst)

    def objective_function(x):
    '''
    Map a sampled domain to evaluation results returned from the model
    :param x: domain sampled from bayesian optimization
    :return: the corresponding evaluation result
    '''
    for i in range(len(rescaled_domain_lst)):
        if (x == rescaled_domain_lst[i]).all():
            return eval_lst[i]

    results, best_ind, fix_budget, close_best = run_bayesian_optimization(args, kernel, objective_function, rescaled_domain_lst, eval_lst, BEST, WORST)
    write_results(args, results, best_ind, fix_budget, close_best)
    

if __name__ == '__main__':
    main()

