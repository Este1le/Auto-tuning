workdir='/export/a08/xzhan138/Auto-tuning/multi-objective/'
output='/export/a08/xzhan138/Auto-tuning/single_output/'

for model_type in "gp" #"gp_mcmc" "rf" "bohamiann" "dngo"
    do
    echo "model_type: ${model_type}"
    for acquisition in "log_ei" # "ei", "lcb" "pi"
        do
        echo "acquisition: ${acquisition}"
        for kernel in "constant" "polynomial" "linear" "dotproduct"
        #for kernel in "exp" "expsquared" "matern32" "matern52"
        #for kernel in "rationalquadratic" "expsine2" "heuristic"
            do
            echo "kernel: ${kernel}"
            for sampling_method in "exact" #"approx"
                do
                echo "sampling_method: ${sampling_method}"
                python ${workdir}bayesian_optimization.py --dataset "robust19-ja-en" \
                                                      --architecture "trans" \
                                                      --rnn-cell-type "lstm" \
                                                      --sampling-method ${sampling_method} \
                                                      --acquisition-func ${acquisition} \
                                                      --model-type ${model_type} \
                                                      --kernel ${kernel} \
                                                      --output ${output}
                done
            done
        done
    done
