import math

base_rescale_dict = {'bpe_symbols': lambda x: (x-10000)/40000, # 10000, 30000, 50000
                     'initial_learning_rate': lambda x: (x-0.0003)/0.0007, # 0.0003, 0.0006, 0.001
                     'num_embed': lambda x: (math.log(x,2)-8)/2 # 256, 512, 1024
}

rnn_rescale_dict = {'num_layers': lambda x: math.log(x,2)/2, # 1, 2, 4
                    'rnn_num_hidden': lambda x: (math.log(x,2)-8)/2, # 256, 512, 1024
                    'batch_size': lambda x: (x-2048)/2048 # 2048, 4096

}

rnn_rescale_dict.update(base_rescale_dict)
rnn_hyps = rnn_rescale_dict.keys()

cnn_rescale_dict = {'num_layers': lambda x: (x-10)/10, # 10, 15, 20
                    'cnn_num_hidden': lambda x: (math.log(x,2)-8)/2, # 256, 512, 1024
                    'batch_size': lambda x: (x-2048)/2048, # 2048, 4096
                    'cnn_kernel_width': lambda x: (x-3)/2 # 3, 5

}

cnn_rescale_dict.update(base_rescale_dict)
cnn_hyps = cnn_rescale_dict.keys()

trans_rescale_dict = {'num_layers': lambda x: (x-2)/2, # 2, 4
                      'transformer_attention_heads': lambda x: (x-8)/8, # 8, 16
                      'transformer_feed_forward_num_hidden': lambda x: (x-1024)/1024, # 1024, 2048
                      'transformer_model_size': lambda x: (math.log(x,2)-8)/2 # 256, 512, 1024
                      # 'batch_size' 4096
}

trans_rescale_dict.update(base_rescale_dict)
trans_hyps = trans_rescale_dict.keys()

def rescale(domain_dict_list, rescale_dict):
    '''
    :param domain_dict_list: A list of domain (hyperparameter names and values) dictionaries.
    :param rescale_dict: Dictionary that defines how to rescale each hyperparameter's values.
    :return: A list of rescaled list.
    '''
    res = []
    for m in domain_dict_list:
        mres = []
        for d in rescale_dict.keys():
            mres.append(rescale_dict[d](m[d]))
        res.append(mres)
    return res


