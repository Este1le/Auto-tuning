"""
Hyperparameter dictionary for Transformer Neural Machine Translation models.
"""

hpm_dict = {}

# Preprocessing
hpm_dict['bpe_symbols'] = [10000, 30000, 50000]

# Model architecture
hpm_dict['num_embed'] = ["\"256:256\"", "\"512:512\"", "\"1024:1024\""]
hpm_dict['num_layers'] = ["\"2:2\"", "\"4:4\""] 

# Training configuration
hpm_dict['batch_size'] = [2048, 4096]
hpm_dict['initial_learning_rate'] = [0.0003, 0.0006, 0.001]

# Transformer
hpm_dict['transformer_model_size'] = [256, 512, 1024]
hpm_dict['transformer_attention_heads'] = [8, 16]
hpm_dict['transformer_feed_forward_num_hidden'] = [1024, 2048]
