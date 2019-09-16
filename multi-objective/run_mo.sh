workdir='/export/a08/xzhan138/Auto-tuning/multi-objective/'
output='/export/a08/xzhan138/Auto-tuning/multi_output/'
seeds=(59 18 20 73 61 29 58 65 14 30)
for dataset in 'ted-zh-en' 'ted-ru-en' 'robust19-en-ja' 'robust19-ja-en'
do
    echo "dataset: ${dataset}"
    for model in 'krr' 'gp' 'gbssl'
    do
        echo "model: ${model}"
        for r in ${seeds[@]}
        do
            echo "random seed: ${r}"
            python ${workdir}multi_optimize.py --dataset ${dataset} \
                                         --model ${model} \
                                         --random-seed ${r} \
                                         --output ${output}
        done
    done
done
