#!/bin/bash

while getopts n:r:v:c:g:m: flag
do
    case "${flag}" in
        n) TRAIN_PATH_NER=${OPTARG};;
        r) TRAIN_PATH_RC=${OPTARG};;
        v) VALIDATION_PATH_NER=${OPTARG};;
        c) VALIDATION_PATH_RC=${OPTARG};;
        g) GPUS=${OPTARG};;
        m) MODEL_PATH=${OPTARG};;
    esac
done


export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES=$GPUS

echo "GPUS:${GPUS}"
echo "MODEL PATH:$MODEL_PATH"

echo "================================================================================="
echo "TRAIN NER MODEL"
echo "================================================================================="

python ../triplet_extraction/train_ner.py \
--train_dataset_path $TRAIN_PATH_NER \
--validation_dataset_path $VALIDATION_PATH_NER \
--model_path $MODEL_PATH \
--output_dir ${OUTPUT}/NER/ \
--lr 0.00003 \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 16 \
--num_train_epochs 3 \
--logging_steps 100 \
--weight_decay 0.01 \
--seed 42 \
--save_steps 500 \
--metric_for_best_model recall 



echo "================================================================================="
echo "TRAIN RC MODEL"
echo "================================================================================="

python ../triplet_extraction/train_rc.py \
--train_dataset_path $TRAIN_PATH_RC \
--validation_dataset_path $VALIDATION_PATH_RC \
--model_path $MODEL_PATH \
--output_dir ${OUTPUT}/RC/ \
--lr 0.00003 \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 16 \
--num_train_epochs 3 \
--logging_steps 100 \
--weight_decay 0.01 \
--seed 42 \
--save_steps 500 \
--metric_for_best_model recall 


echo "================================================================================="
echo "TRAIN QAT MODEL"
echo "================================================================================="
echo "${MODEL_PATH}"
echo "================================================================================="

python ../triplet_extraction/train_qa.py \
--dataset_path sberquad \
--model_path $MODEL_PATH \
--output_dir ${OUTPUT}/QA/ \
--lr 0.00003 \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 16 \
--num_train_epochs 3 \
--logging_steps 100 \
--weight_decay 0.01 \
--seed 42 \
--save_steps 500 \
--metric_for_best_model f1 
