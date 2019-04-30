Scripts for automatically generating hpm files for Neural Machine Translation models trained with Sockeye given possible values of hyperparameters.

## Usage

1. Specify `workdir`, `rootdir`, `src`, `trg`, etc. in `*.hpm.template` files. Notice `???` is a placeholder that will be replaced later with actual values from `*_hpm_dict.py`.

2. Specify hyperparameter values in `*_hpm_dict.py`.

3. Make a directory for generated hpm files, e.g. `rnn_hpm`.

4. Run scripts to generate hpm files:
RNN models:
```bash
python hpm_generator.py -a rnn -o rnn_hpm
```
CNN models:
```bash
python hpm_generator.py -a cnn cnn_hpm
```
Transformer models:
```bash
python hpm_generator.py -a trans trans_hpm
```