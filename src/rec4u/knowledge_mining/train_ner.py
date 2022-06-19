import logging
import sys
import argparse
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    AutoConfig,
)
import torch
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
#from datasets import load_dataset, load_metric
import datasets
from utils.training_utils import set_logger, training_arguments, seed_all
from utils.dataset_er import prepare_ner_data, EntityRecognitionDataset
#from dataset_preparation_ner import EntityRecognitionDataset
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

metric = datasets.load_metric("seqeval")
def compute_metrics(p):
    """ Computes NER metrics in trainer """
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    # Remove ignored index (special tokens)
    true_predictions = [
        [id2tag[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2tag[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
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
        help="Path to the training dataset in TXT format",
    )
    parser.add_argument(
        "--validation_dataset_path",
        default='',
        type=str,
        required=False,
        help="Path to the validation dataset in TXT format",
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
        default='recall',
        type=str,
        required=False,
        help="Metric: one of `recall`, `precision`, `f1`, `accuracy`",
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
   
    logger.info('---- Preparing data ----')
    training_dataset = prepare_ner_data(args.train_dataset_path, tokenizer)
    validation_dataset = prepare_ner_data(args.validation_dataset_path, tokenizer) if args.validation_dataset_path else None

    model = AutoModelForTokenClassification.from_pretrained(args.model_path, num_labels=training_dataset.num_labels)

    if validation_dataset:
        assert training_dataset.num_labels == validation_dataset.num_labels, 'Labels are different in training and validation dataset'

    trainer_args = training_arguments(args, validation_dataset)
    set_logger(trainer_args, logger)

    data_collator = DataCollatorForTokenClassification(tokenizer)
    seed_all(args.seed)
    
    trainer = Trainer(
        model,
        trainer_args,
        train_dataset=training_dataset,
        eval_dataset=validation_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    config = AutoConfig.from_pretrained(
        args.model_path,
        num_labels=training_dataset.num_labels,
        finetuning_task='ner',
        cache_dir=args.output_dir,
    )
    model.config.label2id = training_dataset.tag2id
    model.config.id2label = training_dataset.id2tag
    model.config.save_pretrained(args.output_dir)

    trainer.save_model(args.output_dir)
    logger.info(f'---- Trained model successfully saved into {args.output_dir} and can be used further ----')

if __name__ == '__main__':
    main()
