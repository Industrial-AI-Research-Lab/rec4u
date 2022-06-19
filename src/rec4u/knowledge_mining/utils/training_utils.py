import datasets
import transformers
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
import os
import logging
import sys
import torch
import random
import numpy as np

def seed_all(seed=22):
    """Sets seeds for all random modules"""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)  


def training_arguments(args, validation=None):
    """Initiates the training arguments passed to trainer Huggingface's class"""
    if validation:
        trainer_args = TrainingArguments(
            args.output_dir,
            evaluation_strategy="steps",
            learning_rate=args.lr,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            num_train_epochs=args.num_train_epochs,
            logging_steps=args.logging_steps,
            weight_decay=args.weight_decay,
            lr_scheduler_type='linear',
            seed=args.seed,
            save_steps=args.save_steps,
            load_best_model_at_end=True,
            metric_for_best_model=args.metric_for_best_model,
        )
    else:
        trainer_args = TrainingArguments(
            args.output_dir,
            learning_rate=args.lr,
            per_device_train_batch_size=args.per_device_train_batch_size,
            num_train_epochs=args.num_train_epochs,
            logging_steps=args.logging_steps,
            weight_decay=args.weight_decay,
            lr_scheduler_type='linear',
            seed=args.seed,
            save_steps=args.save_steps,
        )
    return trainer_args

def set_logger(args, logger):
    """Sets logger based on training arguments (args)"""
    log_level = args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
