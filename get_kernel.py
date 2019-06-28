import numpy as np
from george import kernels

trans_domain_kernel_mapping = {"initial_learning_rate": kernels.TransInitialLearningRateKernel,
                               "transformer_attention_heads": kernels.TransAttentionHeadsKernel,
                               "num_layers": kernels.TransNumLayersKernel,
                               "transformer_feed_forward_num_hidden": kernels.TransFeedForwardNumHiddenKernel,
                               "num_embed": kernels.TransNumEmbedKernel,
                               "bpe_symbols": kernels.TransBpeSymbolsKernel,
                               "transformer_model_size": kernels.TransModelSizeKernel}

def get_kernel(architecture, embedding, kernel_name, domain_name_lst):

    if architecture == "trans":
        mapping = trans_domain_kernel_mapping
    else:
        mapping = None

    n_dims = len(domain_name_lst)
    initial_ls = np.ones([n_dims])

    kernel = None

    if embedding == "origin":
        if kernel_name == "constant":
            kernel = kernels.ConstantKernel(1, ndim=n_dims)
        elif kernel_name == "polynomial":
            kernel = kernels.PolynomialKernel(log_sigma2=1, order=3, ndim=n_dims)
        elif kernel_name == "linear":
            kernel = kernels.LinearKernel(log_gamma2=1, order=3, ndim=n_dims)
        elif kernel_name == "dotproduct":
            kernel = kernels.DotProductKernel(ndim=n_dims)
        elif kernel_name == "exp":
            kernel = kernels.ExpKernel(initial_ls, ndim=n_dims)
        elif kernel_name == "expsquared":
            kernel = kernels.ExpSquaredKernel(initial_ls, ndim=n_dims)
        elif kernel_name == "matern32":
            kernel = kernels.Matern32Kernel(initial_ls, ndim=n_dims)
        elif kernel_name == "matern52":
            kernel = kernels.Matern52Kernel(initial_ls, ndim=n_dims)
        elif kernel_name == "rationalquadratic":
            kernel = kernels.RationalQuadraticKernel(log_alpha=1, metric=initial_ls, ndim=n_dims)
        elif kernel_name == "cosine":
            kernel = kernels.CosineKernel(4, ndim=n_dims)
        elif kernel_name == "expsine2":
            kernel = kernels.ExpSine2Kernel(1, 2, ndim=n_dims)
        elif kernel_name == "heuristic":
            kernel = mapping[domain_name_lst[0]](ndim=n_dims, axes=0)
            for i in range(len(domain_name_lst[1:])):
                d = domain_name_lst[1:][i]
                kernel += mapping[d](ndim=n_dims, axes=i)
    elif embedding == "bleu":
        n_dims = 1
        kernel = kernels.ExpSquaredKernel(initial_ls, ndim=n_dims)
    elif embedding == "mds":
        pass

    return kernel

