import os
import sys

def readfile(f):
    '''
    :param f: file path
    :return: list
    '''
    if os.path.isfile(f):
        with open(f) as fobj:
            return fobj.readlines()
    else:
        sys.exit(f + "does not exist.")

def check_converge(model_path):
    '''
    Check if a model is trained to convergence.
    :param model_path: The path to the model directory.
    :return: True or False
    '''
    log_path = os.path.join(model_path, "valid.1best.log")
    if os.path.isfile(log_path):
        with open(log_path) as f:
            for l in f.readlines():
                if "Uncaught exception" in l:
                    return False
        return True
    return False


def isfloat(value):
    '''
    Check whether a string can be converted to float.
    '''
    try:
        float(value)
        return True
    except ValueError:
        return False

def extract_eval_log(model_path):
    '''
    Extract model evaluation results from log file.
    :param model_path: The path to the model directory.
    :return: A dictionary of evaluation results.
    '''
    log_path = os.path.join(model_path, "log")
    lines = readfile(log_path)

    eval_dict = {}
    eval_dict["train_ppl"] = sys.float_info.max

    for l in lines:

        # Number of parameters
        if "Total # of parameters:" in l:
            eval_dict["num_param"] = int(l.strip().split(":")[-1])
        # used cpu time for decoding dev set
        elif "Time-cost" in l:
            #eval_dict["dev_cpu_time"] = float(l.strip().split()[-2].split("=")[-1])
            eval_dict["num_updates"] = int(l.strip().split()[-5].split("=")[-1])
        # gpu memory used for training
        elif "log_gpu_memory_usage" in l:
            eval_dict["gpu_memory"] = int(l.strip().split(":")[-1].split("/")[0])
        # dev perplexity
        elif "Best validation perplexity:" in l:
            eval_dict["dev_ppl"] = float(l.strip().split(":")[-1])
        elif "Train-perplexity=" in l:
            eval_dict["train_ppl"] = min(eval_dict["train_ppl"], float(l.strip().split("=")[-1]))

    return eval_dict

def extract_eval_decode(model_path):
    '''
    Extract model evaluation results from decode files.
    The decoding results should be written under the corresponding model directory.
    :param model_path: The path to the model directory.
    :return: A dictionary of evaluation results.
    '''
    eval_dict = {}
    tb_path = os.path.join(model_path, "test1.1best.bleu")
    tb_log_path = os.path.join(model_path, "test1.1best.log")
    vb_path = os.path.join(model_path, "valid.1best.bleu")
    vb_log_path = os.path.join(model_path, "valid.1best.log")
    # dev BLEU
    eval_dict["dev_bleu"] = float(readfile(vb_path)[0].strip().split()[2][:-1])
    # dev GPU time
    eval_dict["dev_gpu_time"] = float(readfile(vb_log_path)[-3].strip().split()[6][:-1])
    # test BLEU
    eval_dict["test_bleu"] = float(readfile(tb_path)[0].strip().split()[2][:-1])
    # test cpu time
    eval_dict["test_cpu_time"] = float(readfile(tb_log_path)[-3].strip().split()[6][:-1])
    # test perplexity

    return eval_dict

def get_domain(model_name):
    '''
    Get the domain (hyperparameter settings) of a model.
    :param model_path: The path to the model directory.
    :return: A dictionary of the domain.
    '''
    domain_dict = {}

    lst = model_name[6:].split("-")
    domain_dict["architecture"] = lst[0].lower()

    lst = lst[1:]
    for i in range(0, len(lst), 2):
        fst = lst[i]
        snd = lst[i+1].split(":")[0] if ":" in lst[i+1] else lst[i+1].lower()
        domain_dict[fst] = float(snd) if isfloat(snd) else snd

    # Merge bpe_symbols_src and bpe_symbols_trg as they are generated together and always the same
    if "bpe_symbols_src" in domain_dict.keys():
        domain_dict["bpe_symbols"] = domain_dict["bpe_symbols_src"]
        del domain_dict["bpe_symbols_src"]
        del domain_dict["bpe_symbols_trg"]

    return domain_dict

def get_all_domain_eval(models_dir):
    '''
    Get the domain and evaluation results of all the trained models.
    :param models_dir: The directory to all the models.
    :return: [(domain_dict1, eval_dict1), (domain_dict2, eval_dict2), ... ]
    '''
    res = []

    for model_name in os.listdir(models_dir):
        model_path = os.path.join(models_dir, model_name)
        if check_converge(model_path):
            domain_dict = get_domain(model_name)
            eval_dict = extract_eval_log(model_path)
            eval_dict.update(extract_eval_decode(model_path))
            res.append((domain_dict, eval_dict))

    return res

