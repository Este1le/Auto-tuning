qsub -o ~/qsub_log/cm1.log  -l 'hostname=b1[12345678]*|b20|c*,gpu=1,mem_free=12g,ram_free=6g' -V -S /bin/bash -j y ~/sockeye-recipes/scripts/scripts/train.sh -p /export/a16/xzhan138/autotune/hpm/cm1.hpm -e sockeye_gpu

qsub -o ~/qsub_log/cm1_con.log  -l 'hostname=b1[12345678]*|b20|c*,gpu=1,mem_free=12g,ram_free=6g' -V -S /bin/bash -j y ~/sockeye-recipes/scripts/continue-train-cnn.sh -p /export/a16/xzhan138/autotune/hpm/cm1_con.hpm -e sockeye_gpu

qsub -o ~/qsub_log/translate.log -l 'num_proc=2,gpu=1,mem_free=12g,ram_free=6g' -V -S /bin/bash -j y /export/a05/xzhan138/miniscale/wipodicts/xuan/constrained-decode.sh
qsub -o ~/qsub_log/translate.log -l 'num_proc=2,gpu=1,mem_free=12g,ram_free=6g' -V -S /bin/bash -j y /export/a05/xzhan138/miniscale/wipodicts/xuan/standard-decode.sh