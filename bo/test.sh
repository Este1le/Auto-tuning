source /export/a08/xzhan138/Auto-tuning/bayesian_optimization.config

num_run=15
current_time=$(date "+%Y.%m.%d-%H.%M.%S")
for dataset in 'ted-zh-en' 'ted-ru-en' 'robust19-en-ja' 'robust19-ja-en'
do 
	modeldir=/export/a10/kduh/p/mt/gridsearch/${dataset}/models/
	output=${workdir}/testoutput/${current_time}_${dataset}_${architecture}_${rnn_cell_type}_${metric}_${embedding}_${embedding_distance}_${sampling_method}_${replacement}_${acquisition_func}_${model_type}_${kernel}_${weight}.res
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
										   ${weight} \
										   ${replacement}
done