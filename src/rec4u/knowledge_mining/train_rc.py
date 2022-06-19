import torch
import numpy as np
#from sklearn import metrics
import transformers
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from datasets import load_metric
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer
import os
import logging
import sys
import datasets
from utils.training_utils import set_logger, training_arguments, seed_all
from utils.dataset_rc import RelationClassificationDataset, prepare_rc_data
from torch.utils.data import Dataset
import argparse

logger = logging.getLogger(__name__)

accuracy = load_metric("accuracy")
precision = load_metric("precision")
recall = load_metric("recall")
f1 = load_metric("f1")
def _compute_metrics(eval_pred):
    """ Computes classification metrics in trainer """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy_ = accuracy.compute(predictions=predictions, references=labels)
    precision_macro = precision.compute(predictions=predictions, references=labels, average='macro')
    recall_macro = recall.compute(predictions=predictions, references=labels, average='macro')
    f1_macro = f1.compute(predictions=predictions, references=labels, average='macro')
    return {
        'accuracy': accuracy_['accuracy'],
        'precision_macro': precision_macro['precision'],
        'recall_macro': recall_macro['recall'],
        'f1_macro': f1_macro['f1'],
            }


def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_dataset_path",
        type=str,
        required=True,
        help="Path to the training dataset in CSV format w/ `text`, `class` columns",
    )
    parser.add_argument(
        "--validation_dataset_path",
        default='',
        type=str,
        required=False,
        help="Path to the validation dataset in CSV format",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the pretrained model (or model name from Huggingface's hub)",
    )
    parser.add_argument(
        "--output_dir",
        default='.',
        type=str,
        required=False,
        help="Path to save the model checkpoints",
    )
    parser.add_argument(
        "--lr",
        default=3e-5,
        type=float,
        required=False,
        help="learning_rate",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        default=16,
        type=int,
        required=False,
        help="Batch size per device on training",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        default=16,
        type=int,
        required=False,
        help="Batch size per device on validation",
    )
    parser.add_argument(
        "--num_train_epochs",
        default=3,
        type=int,
        required=False,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--logging_steps",
        default=100,
        type=float,
        required=False,
        help="Logging steps for evaluation",
    )
    parser.add_argument(
        "--weight_decay",
        default=0.01,
        type=float,
        required=False,
        help="Weight decay",
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        required=False,
        help="Random seed for model training",
    )
    parser.add_argument(
        "--save_steps",
        default=100,
        type=int,
        required=False,
        help="Steps to make checkpoints",
    )
    parser.add_argument(
        "--metric_for_best_model",
        default='accuracy',
        type=str,
        required=False,
        help="Metric: one of `recall`, `precision`, `f1`, `accuracy`",
    )
    args = parser.parse_args()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, max_len=512)

    training_dataset = prepare_rc_data(args.train_dataset_path, tokenizer)
    validation_dataset = prepare_rc_data(args.validation_dataset_path, tokenizer) if args.validation_dataset_path else None

    model = AutoModelForSequenceClassification.from_pretrained(args.model_path, num_labels=training_dataset.num_labels)

    if validation_dataset:
        assert training_dataset.num_labels == validation_dataset.num_labels, 'Labels are different in training and validation dataset'

    trainer_args = training_arguments(args, validation_dataset)
    set_logger(trainer_args, logger)
    seed_all(args.seed)
    trainer = Trainer(
        model=model,
        args=trainer_args,
        train_dataset=training_dataset,
        eval_dataset=validation_dataset,
        compute_metrics=_compute_metrics,
    )
    trainer.train()

    if training_dataset.tag2id:
        model.config.label2id = training_dataset.tag2id
        model.config.id2label = training_dataset.id2tag
        model.config.save_pretrained(args.output_dir)

    trainer.save_model(args.output_dir)

    logger.info(f'---- Trained model successfully saved into {args.output_dir} and can be used further ----')

if __name__ == '__main__':
    main()
