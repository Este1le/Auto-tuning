import argparse
import numpy as np
from sklearn import linear_model
import preprocess
import rescale

parser = argparse.ArgumentParser(description='Bayesian optimization.')
parser.add_argument('--modeldir', '-d', required=True, default='/export/a10/kduh/p/mt/gridsearch/ted-zh-en/models/',
                    help='The directory to all the models.')
parser.add_argument('--architecture', '-a', required=True, choices=['rnn', 'cnn', 'trans'],
                        help='The architecture of the models to be tuned.')
parser.add_argument('--rnn-cell-type', '-c', required=False, choices=['lstm', 'gru'],
                        help='The cell type of rnn model, required only when the architecture is rnn.')
parser.add_argument('--metric', required=True, choices=['dev_ppl', 'dev_bleu'],
                        help='The objective metric.')

args = parser.parse_args()

domain_eval_lst = preprocess.get_all_domain_eval(args.modeldir)
if args.architecture == 'rnn':
    domain_eval_lst_arch = [de for de in domain_eval_lst if (de[0]['architecture']==args.architecture) and (de[0]['rnn_cell_type']==args.rnn_cell_type)]
else:
    domain_eval_lst_arch = [de for de in domain_eval_lst if (de[0]['architecture']==args.architecture)]
domain_dict_lst, eval_dict_lst = list(list(zip(*domain_eval_lst_arch))[0]), list(list(zip(*domain_eval_lst_arch))[1])
rescaled_domain_lst, domain_name_lst = rescale.rescale(domain_dict_lst, rescale.trans_rescale_dict)
eval_lst = [e[args.metric] for e in eval_dict_lst]

x = np.array(rescaled_domain_lst)
y = np.array(eval_lst)
regr = linear_model.LinearRegression()
regr.fit(x,y)
c = regr.coef_
cscaled = [abs(i) for i in c]
minc = min(cscaled)
maxc = max(cscaled)
cscaled = [0.9-(i-minc)/(maxc-minc)*0.8 for i in cscaled]

print("Domain names: ", domain_name_lst)
print("Coeficients: ", c)
print("Scaled Coeficients [0.1, 0.9]: ", cscaled)