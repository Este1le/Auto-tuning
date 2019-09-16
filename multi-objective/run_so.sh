workdir='/export/a08/xzhan138/Auto-tuning/multi-objective/'
output='/export/a08/xzhan138/Auto-tuning/single_output/'
acquisition='max'
cell='lstm'
for architecture in 'rnn' 'trans'
    do
    echo "architecture: ${architecture}"
    echo "rnn_cell_type: ${cell}"
    for dataset in 'ted-zh-en' 'ted-ru-en' 'robust19-en-ja' 'robust19-ja-en'
        do
        echo "dataset: ${dataset}"
        for model in 'krr' 'gp' 'gbssl'
            do
            echo "model: ${model}"
            python ${workdir}single_optimize.py --dataset ${dataset} \
                                                --architecture ${architecture} \
                                                --rnn-cell-type ${cell} \
                                                --model ${model} \
                                                --acquisition ${acquisition} \
                                                --output ${output}
            done
    done
done
