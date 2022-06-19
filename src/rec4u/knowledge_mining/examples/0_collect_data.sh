#!/bin/bash
while getopts t:d:r:o:g:t flag
do
    case "${flag}" in
        t) TEXTS=${OPTARG};;
        d) ENTITIES=${OPTARG};;
        r) RELATIONS=${OPTARG};;
        o) OUTPUT=${OPTARG};;
        g) GPUS=${OPTARG};;
        t) TOPN=${OPTARG};;
       
    esac
done

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES=$GPUS

echo "GPUS:${GPUS}"

echo "================================================================================="
echo "COLLECTING NER DATA"
echo "================================================================================="

python triplet_extraction/text2tags_ner.py \
--texts_path ${TEXTS} \
--dict_path ${ENTITIES} \
--output_dir ${OUTPUT}

echo "================================================================================="
echo "COLLECTING RC DATA"
echo "================================================================================="

python triplet_extraction/text2lbls_relclf.py \
--relation_examples ${RELATIONS} \
--texts_path ${TEXTS} \
--output_dir ${OUTPUT} \
--top_n ${TOPN} \
--range_threshold 0.3 

