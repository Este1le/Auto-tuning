import argparse
import logging
import random
import numpy as np
import preprocess

logging.basicConfig(level=logging.INFO)

def get_args():
    parser = argparse.ArgumentParser(description='Bayesian optimization.')

    # Model and Architecture arguments
    parser.add_argument('--modeldir', '-d', required=True, default='/export/a10/kduh/p/mt/gridsearch/ted-zh-en/models/',
                        help='The directory to all the models.')
    parser.add_argument('--architecture', '-a', required=True, choices=['rnn', 'cnn', 'trans'],
                        help='The architecture of the models to be tuned.')

    # Objective Metric arguments
    parser.add_argument('--metric', required=True, choices=['dev_ppl', 'dev_bleu'],
                        help='The objective metric.')
    parser.add_argument('--best', type=str, required=True, choices=['min', 'max'],
                        help='How should we get the best evaluation result in the pool? By min or max?')

    # Random Search arguments
    parser.add_argument('--num-run', type=int, default=100,
                        help='Number of random search runs.')

    # Evaluation arguments
    parser.add_argument('--budget', type=int, default=10,
                        help='This is for the use of measuring the performance of random search given limited budget.\
                        Or the number of models we are allowed to evaluate. \
                        Or the number of iterations of random search we are allowed to run.')
    parser.add_argument('--dif', type=float, default=1,
                        help='This is for the use of checking how many iterations should random search take to\
                        to achieve a result whose difference to the best result is no larger than dif.')

    # Output arguments
    parser.add_argument('--output', required=True, help='The output path.')

    args = parser.parse_args()

    return args

def extract_data(args):
    logging.info("Extracting domains and evaluation results for all models")
    domain_eval_lst = preprocess.get_all_domain_eval(args.modeldir)
    domain_eval_lst_arch = [de for de in domain_eval_lst if (de[0]['architecture']==args.architecture)]
    domain_dict_lst, eval_dict_lst = list(list(zip(*domain_eval_lst_arch))[0]), list(list(zip(*domain_eval_lst_arch))[1])

    # The objective we want to optimize
    if args.best == 'min':
        eval_lst = [e[args.metric] for e in eval_dict_lst]
    else:
        eval_lst = [-e[args.metric] for e in eval_dict_lst]
    BEST = min(eval_lst)

    return eval_lst, BEST

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

def run_random_search(args, eval_lst, BEST):
    results = []
    best_ind = []
    fix_budget = []
    close_best = []
    origin_eval_lst = eval_lst[:]

    for i in range(args.num_run):
        sampled_id = []
        invs = []
        eval_lst = origin_eval_lst[:]

        logging.info("# %d run of random search.", i)

        while BEST not in invs:
            n = random.choice(range(len(eval_lst)))
            invs.append(eval_lst[n])
            eval_lst = eval_lst[:n] + eval_lst[n+1:]

        logging.info("%d number of iterations to get the best.\n", len(invs))

        results.append(invs)

        best_ind.append(len(invs))

        fix_budget.append(abs(min(invs[:args.budget])-BEST))

        for i in range(len(invs)):
            if invs[i]-BEST <= args.dif:
                close_best.append(i+1)
                break

    return results, best_ind, fix_budget, close_best

def main():
    args = get_args()
    eval_lst, BEST = extract_data(args)

    # shuffle the data
    random.Random(37).shuffle(eval_lst)

    results, best_ind, fix_budget, close_best = run_random_search(args, eval_lst, BEST)
    write_results(args, results, best_ind, fix_budget, close_best)
    

if __name__ == '__main__':
    main()