"""
Hyperparameter dictionary for CNN-based Neural Machine Translation models.
"""

hpm_dict = {}

# Preprocessing
hpm_dict['bpe_symbols'] = [10000, 30000, 50000]

# Model architecture
hpm_dict['num_embed'] = ["\"256:256\"", "\"512:512\"", "\"1024:1024\""]
hpm_dict['num_layers'] = ["\"10:10\"", "\"15:15\"", "\"20:20\""]

# Training configuration
hpm_dict['batch_size'] = [2048, 4096]
hpm_dict['initial_learning_rate'] = [0.0003, 0.0006, 0.001]

# CNN
hpm_dict['cnn_num_hidden'] = [256, 512, 1024]
hpm_dict['cnn_kernel_width'] = ["\"3:3\"", "\"5:5\""]
