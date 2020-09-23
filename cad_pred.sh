#!/bin/bash
EPOCHS=$1
LOG_PATH="/media/hdd/home/baoxiong/Projects/TPAMI2019/tmp/cad/log/nn_results"
pred_duration=("15" "30" "45" "60" "75" "90" "105" "120" "135" "150")

if [ ! -d ${LOG_PATH} ]
then
    mkdir ${LOG_PATH}
fi

for pred in "${!pred_duration[@]}"
do
	python experiments/LSTM/pred_baseline.py --task activity --dataset CAD --batch_size 1 --lr 1e-3 --lr_decay 0.8 --epochs ${EPOCHS} --pred_duration ${pred_duration[$pred]} > ${LOG_PATH}/pred${pred_duration[$pred]}_train.txt
done

for pred in "${!pred_duration[@]}"
do
	python experiments/LSTM/pred_baseline.py --task activity --dataset CAD --batch_size 1 --lr 5e-4 --lr_decay 0.8 --epochs ${EPOCHS} --pred_duration ${pred_duration[$pred]} --eval True > ${LOG_PATH}/pred${pred_duration[$pred]}_eval.txt
done

for pred in "${!pred_duration[@]}"
do
	python experiments/GEP/gep_pred_topdown.py --task activity --dataset CAD --batch_size 1 --lr 5e-4 --lr_decay 0.8 --epochs ${EPOCHS} --pred_duration ${pred_duration[$pred]} --using_pred_duration ${pred_duration[$pred]} > ${LOG_PATH}/gep_pred${pred_duration[$pred]}_eval.txt
done
