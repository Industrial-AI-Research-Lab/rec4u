#!/bin/bash
while getopts t:r:e:n:c:q:o: flag
do
    case "${flag}" in
        t) TEXT_PATH=${OPTARG};;
        r) RELATION_TEMPLATES=${OPTARG};;
        e) ENTITY_ALIASES=${OPTARG};;
        n) NER_MODEL_PATH=${OPTARG};;
        c) RC_MODEL_PATH=${OPTARG};;
        q) QAT_MODEL_PATH=${OPTARG};;
        o) OUTPUT_PATH=${OPTARG};;
    esac
done

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES=$GPUS

python ../triplet_extraction/extract_triples.py \
--texts_path $TEXT_PATH \
--relation_templates $RELATION_TEMPLATES \
--entity_aliases $ENTITY_ALIASES \
--ner_model_path $NER_MODEL_PATH \
--rc_model_path $RC_MODEL_PATH \
--qat_model_path $QAT_MODEL_PATH \
--output_path $OUTPUT_PATH \
--return_unmerged_triples False 