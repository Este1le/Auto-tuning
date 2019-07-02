source /export/a08/xzhan138/Auto-tuning/random_search.config

python ${workdir}/random_search.py --modeldir ${modeldir} \
								   --architecture ${architecture} \
								   --metric ${metric} \
								   --best ${best} \
								   --budget ${budget} \
								   --dif ${dif} \
								   --output ${output}