#!/bin/bash
EPOCHS=$1
LOG_PATH="media/hdd/home/baoxiong/Projects/TPAMI2019/tmp/breakfast/log/nn_results"
subsample=("1" "2" "5" "10" "20" "50" "100")
batch_size=("20" "32" "32" "32" "32" "32" "32")
training_epochs=("5" "10" "15" "20" "25" "30" "35" "40" "45" "50")

if [ ! -d ${LOG_PATH} ]
then
    mkdir ${LOG_PATH}
fi

for sub in "${!subsample[@]}"
do
  	python experiments/LSTM/detect.py --task activity --dataset Breakfast --batch_size ${batch_size[$sub]} --lr 1e-3 --lr_decay 0.8 --epochs ${EPOCHS} --subsample ${subsample[$sub]} --save_interval 5 > ${LOG_PATH}/s${subsample[$sub]}_b${batch_size[$sub]}.txt
done

for sub in "${!subsample[@]}"
do
    for trainepochs in "${training_epochs[@]}"
    do
        python experiments/LSTM/detect.py --task activity --dataset Breakfast --batch_size ${batch_size[$sub]} --lr 1e-3 --lr_decay 0.8 --epochs ${EPOCHS} --subsample ${subsample[$sub]} --save_interval 5 --trained_epochs ${trainepochs} --eval True > ${LOG_PATH}/eval_s${subsample[$sub]}_b${batch_size[$sub]}_t${trainepochs}.txt
    done
done
