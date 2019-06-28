source /export/a08/xzhan138/Auto-tuning/bayesian_optimization.config

python ${workdir}/bayesian_optimization.py --modeldir ${modeldir} \
										   --architecture ${architecture} \
										   --rnn-cell-type ${rnn_cell_type} \
								           --metric ${metric} \
								           --embedding ${embedding} \
								           --embedding-distance ${embedding_distance} \
										   --best ${best} \
										   --sampling-method ${sampling_method} \
										   --acquisition-func ${acquisition_func} \
										   --model-type ${model_type} \
										   --num-run ${num_run} \
										   --kernel ${kernel} \
										   --budget ${budget} \
										   --dif ${dif} \
										   --output ${output} \
										   ${replacement} 