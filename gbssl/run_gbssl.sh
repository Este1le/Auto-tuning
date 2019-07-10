source /export/a08/xzhan138/Auto-tuning/gbssl/gbssl.config

python ${workdir}/gbssl.py --modeldir ${modeldir} \
						   --architecture ${architecture} \
						   --rnn-cell-type ${rnn_cell_type} \
						   --metric ${metric} \
						   --best ${best} \
						   --distance ${distance} \
						   --sparsity ${sparsity} \
						   --k ${k} \
						   --budget ${budget} \
						   --dif ${dif} \
						   --num-run ${num_run} \
						   --output ${output}

