#!/bin/bash
TRAINED_EPOCHS=$1

LOG_PATH="/media/hdd/home/baoxiong/Projects/TPAMI2019/tmp/breakfast/log/gep_results"

subsample=("1" "2" "5" "10" "20" "50")
batch_size=("20" "32" "32" "32" "32" "32")

if [ ! -d ${LOG_PATH} ]
then
    mkdir ${LOG_PATH}
fi

for subs in "${!subsample[@]}"
do
    echo GEP_${subsample[$subs]}_b${batch_size[$subs]}_t${TRAINED_EPOCHS}
    python experiments/GEP/gep.py --task activity --dataset Breakfast --using_batch_size ${batch_size[$subs]} --subsample ${subsample[$subs]} --lr 1e-3 --lr_decay 0.8 --epochs 50 --trained_epochs ${TRAINED_EPOCHS} > ${LOG_PATH}/eval_s${subsample[$subs]}_b${batch_size[$subs]}_t${TRAINED_EPOCHS}.txt
done
