source /export/a08/xzhan138/Auto-tuning/gbssl/gbssl.config

python ${workdir}/gbssl.py --modeldir ${modeldir} \
						   --architecture ${architecture} \
						   --rnn-cell-type ${rnn_cell_type} \
						   --metric ${metric} \
						   --distance ${distance} \
						   --distance-param ${distance_param} \
						   --sparsity ${sparsity} \
						   --k ${k}

