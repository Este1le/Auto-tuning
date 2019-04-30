"""
Generate hyperparameter files for Neural Machine Translation models trained with Sockeye.
"""

import os
import argparse
import rnn_hpm_dict
import cnn_hpm_dict
import trans_hpm_dict

def get_combs(hpm_dict):
    '''
    get all the combinations of hyperparameter settings.
    '''
    hpm_combs = [{}]
    for h in hpm_dict.keys():
        v_lst = hpm_dict[h]
        t_combs = []
        for v in v_lst:
            for d in hpm_combs:
                newd = dict(d)
                newd[h] = v
                t_combs.append(newd)
        hpm_combs = t_combs
    return hpm_combs

def convert_bpe(h_dict):
    '''
    Change bpe_symbols to bpe_symbols_src and bpe_symbols_trg.
    '''
    h_dict["bpe_symbols_src"] = h_dict["bpe_symbols"]
    h_dict["bpe_symbols_trg"] = h_dict["bpe_symbols"]
    del h_dict["bpe_symbols"]
    return h_dict

def dict2str(h_dict):
    '''
    Convert hyperparameter dictionary to a string for hpm file name and model name.
    '''
    s = "{0}".format(arch.upper())
    for h in h_dict.keys():
        s += "-{0}-{1}".format(h, str(h_dict[h]).replace("\"",""))
    return s

def write_hpm(hpm_tmp, hpm_combs, outd):
    '''
    Write hyperparameter values to hpm files. 
    '''
    with open(hpm_tmp) as f:
        lines = f.readlines()

    for h_dict in hpm_combs:
        h_dict = convert_bpe(h_dict)
        newlines = list(lines)
        # Replace symbol with actual values
        for i in range(len(newlines)):
            l = newlines[i]
            if symbol in l:
                if "modeldir" in l:
                    newlines[i] = l.replace(symbol, dict2str(h_dict))
                elif l.split("=")[0] in h_dict:
                    newlines[i] = l.replace(symbol, str(h_dict[l.split("=")[0]]))

        # Write to hpm file
        with open(os.path.join(outd, dict2str(h_dict)+".hpm"), "w") as f:
            f.write("".join(newlines))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate hyperparameter files for Neural Machine Translaion models.')

    parser.add_argument('--architecture', '-a', choices=['rnn', 'cnn', 'trans'],
                        help='Architecture of the NMT model: RNN, CNN or Transformer.')
    parser.add_argument('--output', '-o', default='./', help='The output directory.')

    args = parser.parse_args()
    arch = args.architecture
    outd = args.output

    symbol = "???" # symbol for unknown hyperparameter values in the hpm template files

    hpm_dict = eval("{0}_hpm_dict".format(arch)).hpm_dict
    hpm_tmp = "{0}.hpm.template".format(arch)

    hpm_combs = get_combs(hpm_dict)
    print("Numbers of {0} models to generate: {1}.".format(arch, len(hpm_combs)))

    print("Writing to files...")
    write_hpm(hpm_tmp, hpm_combs, outd)
    print("Done")





