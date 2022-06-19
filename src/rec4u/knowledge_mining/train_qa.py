from datasets import load_dataset, load_metric
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    default_data_collator,
)
import transformers
from utils.train_qa_utils import (
    prepare_train_features,
    prepare_validation_features,
    postprocess_qa_predictions,
    squad_compute
)
import logging
import sys
import argparse
from utils.training_utils import set_logger, training_arguments, seed_all
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

logger = logging.getLogger(__name__)

def compute_metrics(raw_predictions, datasets, validation_features, args):
    """ Computes question-answering metrics in trainer """
    final_predictions = postprocess_qa_predictions(datasets['validation'], \
                                                   validation_features, raw_predictions.predictions,
                                                   model_checkpoint=args.model_path)
    formatted_predictions = [{"id": k, "prediction_text": v, "no_answer_probability": 0.0}
                             for k, v in final_predictions.items()]
    references = [{"id": ex["id"], "answers": ex["answers"]}
                  for ex in datasets['validation']]
    res, ind, f1 = squad_compute(predictions=formatted_predictions, references=references, print_over=0.2)
    return res

def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        default='sberquad',
        type=str,
        required=False,
        help="Path to the dataset in SberQuAD format (or dataset name from Huggingface's hub)",
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
        default='f1',
        type=str,
        required=False,
        help="Metric: one of `exact_match`, `f1`",
    )
    args = parser.parse_args()

    datasets = load_dataset(args.dataset_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model_path)
    logger.info('Training data tokenization')
    tokenized_datasets = datasets.map(
        lambda x: prepare_train_features(x, args.model_path),
        batched=True,
        remove_columns=datasets["train"].column_names
    )
    logger.info('Preparing validation features')
    validation_features = datasets["validation"].map(
        lambda x: prepare_validation_features(x, args.model_path),
        batched=True,
        remove_columns=datasets["validation"].column_names
    )

    trainer_args = training_arguments(args, validation=True)
    set_logger(trainer_args, logger)
    data_collator = default_data_collator
    seed_all(args.seed)
    trainer = Trainer(
        model,
        trainer_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=lambda x: compute_metrics(x, datasets, validation_features, args),
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    logger.info(f'---- Trained model successfully saved into {args.output_dir} and can be used further ----')

if __name__ == '__main__':
    main()
