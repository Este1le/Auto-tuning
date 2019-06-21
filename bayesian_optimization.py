import argparse
import logging
import numpy as np
from sklearn import metrics
from robo.fmin import bayesian_optimization
import preprocess
import rescale

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(description='Bayesian optimization.')
parser.add_argument('--modeldir', '-d', default='/export/a10/kduh/p/mt/gridsearch/ted-zh-en/models/',
                    help='The directory to all the models.')
parser.add_argument('--architecture', '-a', required=True, choices=['rnn', 'cnn', 'trans'],
                    help='The architecture of the models to be tuned.')
parser.add_argument('--rnn-cell-type', '-c', required=False, choices=['lstm', 'gru'])

args = parser.parse_args()

models_dir = args.modeldir
arch = args.architecture
if arch=='rnn' and 'rnn_cell_type' not in vars(args):
    parser.error('--rnn-cell-type is required if the architecture is rnn.')

rnn_cell_type = args.rnn_cell_type

logging.info("Extracting domains and evaluation results for all models")
domain_eval_lst = preprocess.get_all_domain_eval(models_dir)
domain_eval_lst_arch = [de for de in domain_eval_lst if (de[0]['architecture']==arch) and (de[0]['rnn_cell_type']==rnn_cell_type)]

domain_dict_lst, eval_dict_lst = list(list(zip(*domain_eval_lst_arch))[0]), list(list(zip(*domain_eval_lst_arch))[1])

# Rescale values to the range [0,1] and turn dictionary into list
if arch=='rnn':
    rescaled_domain_lst = rescale.rescale(domain_dict_lst, rescale.rnn_rescale_dict)
elif arch=='cnn':
    rescaled_domain_lst = rescale.rescale(domain_dict_lst, rescale.cnn_rescale_dict)
elif arch=='trans':
    rescaled_domain_lst = rescale.rescale(domain_dict_lst, rescale.trans_rescale_dict)

# The objective we want to optimize
eval_lst = [e['dev_ppl'] for e in eval_dict_lst]
origin_rdl = rescaled_domain_lst[:]
origin_el = eval_lst[:]

def objective_function(x):
    '''
    Map a sampled domain to evaluation results returned from the model
    :param x: domain sampled from bayesian optimization
    :return: the corresponding evaluation result
    '''
    # max = 0
    # max_id = 0
    # for i in range(len(rescaled_domain_lst)):
    #     x_ = rescaled_domain_lst[i]
    #     # Choose a domain that is most similar (measured by cosine similarity) to the sampled vector
    #     cos_sim = abs(metrics.pairwise.cosine_similarity(np.array(x).reshape(1,-1), np.array(x_).reshape(1,-1))[0][0])
    #     if cos_sim > max:
    #         max = cos_sim
    #         max_id = i
    # return eval_lst[max_id]
    for i in range(len(rescaled_domain_lst)):
        if (x == rescaled_domain_lst[i]).all():
            return eval_lst[i]

lower = np.array([0]*len(rescaled_domain_lst[0]))
upper = np.array([1]*len(rescaled_domain_lst[0]))

results = []
best_ind = []
fix_budget = []
close_best = []

BG = 10
DIF = 1
num_run = 100

BEST = min(eval_lst) # max for BLEU
WORST = 100000 # 0 for BLEU
kernel = "matern52"
sampling_method = "exact"
replacement = False

for _ in range(num_run):
    rescaled_domain_lst = origin_rdl[:]
    eval_lst = origin_el[:]
    logging.info("#" + str(_) + " run of bayesian optimization.")
    result = bayesian_optimization(objective_function, lower,upper, acquisition_func="log_ei", model_type="gp", 
                                   num_iterations=len(origin_rdl), kernel=kernel, sampling_method=sampling_method, 
                                   replacement=replacement, pool=np.array(rescaled_domain_lst), best=BEST)
    results.append(result)
    invs = result["incumbent_values"]

    if BEST in invs:
        best_ind.append(invs.index(BEST)+1)
    else:
        best_ind.append(WORST)

    fix_budget.append(abs(min(invs[:BG])-BEST))

    for i in range(len(invs)):
        if invs[i]-BEST <= DIF:
            close_best.append(i+1)
            break

if replacement:
    rep = "rep"
else:
    rep = "norep"

with open("/export/a08/xzhan138/Auto-tuning/bo_" + kernel + "_" + sampling_method + "_" + rep + ".res", "w") as f:
    f.write(str(results))
    f.write("\n")
    f.write("########################################\n\n")
    f.write("Best ind\n")
    f.write(str(best_ind) + "\n")
    f.write("Ave: " + str(sum(best_ind)/len(best_ind)) + "\n")
    f.write("Var: " + str(np.std(best_ind)) + "\n\n")

    f.write("Fix budget(10)\n")
    f.write(str(fix_budget) + "\n")
    f.write("Ave: " + str(sum(fix_budget)/len(fix_budget)) + "\n")
    f.write("Var: " + str(np.std(fix_budget)) + "\n\n")

    f.write("Close best(1) \n")
    f.write(str(close_best) + "\n")
    f.write("Ave: " + str(sum(close_best)/len(close_best)) + "\n")
    f.write("Var: " + str(np.std(close_best)))
