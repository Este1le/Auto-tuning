import numpy as np
from george import kernels

trans_domain_kernel_mapping = {"initial_learning_rate": kernels.TransInitialLearningRateKernel,
                               "transformer_attention_heads": kernels.TransAttentionHeadsKernel,
                               "num_layers": kernels.TransNumLayersKernel,
                               "transformer_feed_forward_num_hidden": kernels.TransFeedForwardNumHiddenKernel,
                               "num_embed": kernels.TransNumEmbedKernel,
                               "bpe_symbols": kernels.TransBpeSymbolsKernel,
                               "transformer_model_size": kernels.TransModelSizeKernel}

def get_kernel(architecture, embedding_name, embedding_distance, kernel_name, domain_name_lst, n_dims, weight):

    if architecture == "trans":
        mapping = trans_domain_kernel_mapping
    else:
        mapping = None

    kernel = None
    initial_ls = np.ones([n_dims])

    if embedding_name == "mds":
        if embedding_distance == "heuristic":
            embedding_name = "origin"
            kernel_name = "heuristic"
        elif embedding_distance == "bleudif":
            embedding_name = "bleu"

    if embedding_name == "origin":
        if not weight:
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
            elif kernel_name == "weightheuristic":
                kernel = mapping[domain_name_lst[0]](ndim=n_dims, axes=0) \
                         * kernels.ExpSquaredKernel(metric=1, ndim=n_dims, axes=0)
                for i in range(len(domain_name_lst[1:])):
                    d = domain_name_lst[1:][i]
                    kernel += mapping[d](ndim=n_dims, axes=i) \
                              * kernels.ExpSquaredKernel(metric=1, ndim=n_dims, axes=i)
            elif kernel_name == "logsquared":
                kernel = kernels.LogSquaredKernel(initial_ls, ndim=n_dims)
        else:
            if kernel_name == "constant":
                kernel = mapping[domain_name_lst[0]](ndim=n_dims, axes=0)
                for i in range(len(domain_name_lst[1:])):
                    d = domain_name_lst[1:][i]
                    kernel += mapping[d](ndim=n_dims, axes=i)

            elif kernel_name == "polynomial":
                kernel = mapping[domain_name_lst[0]](ndim=n_dims, axes=0) \
                         * kernels.PolynomialKernel(log_sigma2=1, order=3, ndim=n_dims, axes=0)
                for i in range(len(domain_name_lst[1:])):
                    d = domain_name_lst[1:][i]
                    kernel += mapping[d](ndim=n_dims, axes=i) \
                              * kernels.PolynomialKernel(log_sigma2=1, order=3, ndim=n_dims, axes=i)

            elif kernel_name == "linear":
                kernel = mapping[domain_name_lst[0]](ndim=n_dims, axes=0) \
                         * kernels.LinearKernel(log_gamma2=1, order=3, ndim=n_dims, axes=0)
                for i in range(len(domain_name_lst[1:])):
                    d = domain_name_lst[1:][i]
                    kernel += mapping[d](ndim=n_dims, axes=i) \
                              * kernels.LinearKernel(log_gamma2=1, order=3, ndim=n_dims, axes=i)

            elif kernel_name == "dotproduct":
                kernel = mapping[domain_name_lst[0]](ndim=n_dims, axes=0) \
                         * kernels.DotProductKernel(ndim=n_dims, axes=0)
                for i in range(len(domain_name_lst[1:])):
                    d = domain_name_lst[1:][i]
                    kernel += mapping[d](ndim=n_dims, axes=i) \
                              * kernels.DotProductKernel(ndim=n_dims, axes=i)

            elif kernel_name == "exp":
                kernel = mapping[domain_name_lst[0]](ndim=n_dims, axes=0) \
                         * kernels.ExpKernel(metric=1, ndim=n_dims, axes=0)
                for i in range(len(domain_name_lst[1:])):
                    d = domain_name_lst[1:][i]
                    kernel += mapping[d](ndim=n_dims, axes=i) \
                              * kernels.ExpKernel(metric=1, ndim=n_dims, axes=i)

            elif kernel_name == "expsquared":
                kernel = mapping[domain_name_lst[0]](ndim=n_dims, axes=0) \
                         * kernels.ExpSquaredKernel(metric=1, ndim=n_dims, axes=0)
                for i in range(len(domain_name_lst[1:])):
                    d = domain_name_lst[1:][i]
                    kernel += mapping[d](ndim=n_dims, axes=i) \
                              * kernels.ExpSquaredKernel(metric=1, ndim=n_dims, axes=i)

            elif kernel_name == "matern32":
                kernel = mapping[domain_name_lst[0]](ndim=n_dims, axes=0) \
                         * kernels.Matern32Kernel(1, ndim=n_dims, axes=0)
                for i in range(len(domain_name_lst[1:])):
                    d = domain_name_lst[1:][i]
                    kernel += mapping[d](ndim=n_dims, axes=i) \
                              * kernels.Matern32Kernel(1, ndim=n_dims, axes=i)

            elif kernel_name == "matern52":
                kernel = mapping[domain_name_lst[0]](ndim=n_dims, axes=0) \
                         * kernels.Matern52Kernel(1, ndim=n_dims, axes=0)
                for i in range(len(domain_name_lst[1:])):
                    d = domain_name_lst[1:][i]
                    kernel += mapping[d](ndim=n_dims, axes=i) \
                              * kernels.Matern52Kernel(1, ndim=n_dims, axes=i)

            elif kernel_name == "rationalquadratic":
                kernel = mapping[domain_name_lst[0]](ndim=n_dims, axes=0) \
                         * kernels.RationalQuadraticKernel(log_alpha=1, metric=1, ndim=n_dims, axes=0)
                for i in range(len(domain_name_lst[1:])):
                    d = domain_name_lst[1:][i]
                    kernel += mapping[d](ndim=n_dims, axes=i) \
                              * kernels.RationalQuadraticKernel(log_alpha=1, metric=1, ndim=n_dims, axes=i)

            elif kernel_name == "cosine":
                kernel = mapping[domain_name_lst[0]](ndim=n_dims, axes=0) \
                         * kernels.CosineKernel(4, ndim=n_dims, axes=0)
                for i in range(len(domain_name_lst[1:])):
                    d = domain_name_lst[1:][i]
                    kernel += mapping[d](ndim=n_dims, axes=i) \
                              * kernels.CosineKernel(4, ndim=n_dims, axes=i)

            elif kernel_name == "expsine2":
                kernel = mapping[domain_name_lst[0]](ndim=n_dims, axes=0) \
                         * kernels.ExpSine2Kernel(1, 2, ndim=n_dims, axes=0)
                for i in range(len(domain_name_lst[1:])):
                    d = domain_name_lst[1:][i]
                    kernel += mapping[d](ndim=n_dims, axes=i) \
                              * kernels.ExpSine2Kernel(1, 2, ndim=n_dims, axes=i)

            elif kernel_name == "heuristic":
                kernel = mapping[domain_name_lst[0]](ndim=n_dims, axes=0)
                for i in range(len(domain_name_lst[1:])):
                    d = domain_name_lst[1:][i]
                    kernel += mapping[d](ndim=n_dims, axes=i)
                
            elif kernel_name == "logsquared":
                kernel = kernels.LogSquaredKernel(initial_ls, ndim=n_dims)

    elif embedding_name == "bleu":
        kernel = kernels.ExpSquaredKernel(initial_ls, ndim=n_dims)

    elif embedding_name == "ml":
        initial_ls = np.ones([n_dims])*100
        kernel = kernels.ExpSquaredKernel(initial_ls, ndim=n_dims)


    return kernel

