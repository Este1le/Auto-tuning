source /export/a08/xzhan138/Auto-tuning/gbssl/gbssl.config

num_run=15
current_time=$(date "+%Y.%m.%d-%H.%M.%S")
for dataset in 'ted-zh-en' 'ted-ru-en' 'robust19-en-ja' 'robust19-ja-en'
do
    modeldir=/export/a10/kduh/p/mt/gridsearch/${dataset}/models/
    for sparsity in 'full' 'knn'
    do
        output=${workdir}/testoutput/${current_time}_${dataset}_${architecture}_${rnn_cell_type}_${model}_${metric}_${distance}_${distance_param}_${sparsity}_${k}.res
        python ${workdir}/gbssl.py --modeldir ${modeldir} \
                   --architecture ${architecture} \
                   --rnn-cell-type ${rnn_cell_type} \
                   --metric ${metric} \
                   --best ${best} \
                   --model ${model} \
                   --distance ${distance} \
                   --sparsity ${sparsity} \
                   --k ${k} \
                   --budget ${budget} \
                   --dif ${dif} \
                   --num-run ${num_run} \
                   --output ${output}
    done
done

