from transformers import (
    BertForTokenClassification,
    BertForSequenceClassification,
    BertForQuestionAnswering,
    BertTokenizer,
    pipeline,
)
import torch
import sys
import transformers
from razdel import sentenize
from typing import Sequence, Union
import pandas as pd
import logging
import argparse
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import os
import json
logger = logging.getLogger(__name__)
from typing import Sequence, Union

def _process_res_ner(res_ner: Sequence[Union[None, dict]]):
    """ Groups found entities by `Label` """
    found_entities = {}
    for ent in sorted(res_ner, key=lambda x: x['score']):
        if ent['entity_group'] not in found_entities.keys():
            found_entities[ent['entity_group']] = []
        found_entities[ent['entity_group']].append({
            'word': ent['word'],
            'score': ent['score'],
        })
    return found_entities

def _classifyer(
    text: str,
    clf_pipeline: transformers.pipelines.text_classification.Pipeline
    ):
    """ Manages tokenization errors in classification pipeline """
    try:
        return clf_pipeline(text)
    except RuntimeError:
        return -1

def _create_questions(row: pd.core.series.Series, relation_examples: Sequence[str], lookup: dict):
    """ Creates question for QAT model from templates """
    return f'{relation_examples[row.relation_classes]} {lookup[row.entity_class]}'

def _read_and_prepare_data(path: str):
    """ Loads texts containing cadidate triples and processes them in applicable for the method view """
    with open(path) as f:
        texts = f.readlines()
    df = pd.DataFrame({'text':texts})
    tqdm.pandas(desc="Splitting texts into sents...")
    with logging_redirect_tqdm():
        df['sents'] = df.progress_apply(lambda x: [_.text for _ in sentenize(x['text']) ], axis=1)
        df = df.explode('sents')
    return df

def _initialize_pipelines(path_er_model: str, path_rc_model: str, path_qa_model: str):
    """ Loads pretrained models and tokenizer, initiates the pipelines """
    model_er = BertForTokenClassification.from_pretrained(path_er_model)
    model_rc = BertForSequenceClassification.from_pretrained(path_rc_model)
    model_qa = BertForQuestionAnswering.from_pretrained(path_qa_model)

    tokenizer = BertTokenizer.from_pretrained(
        path_er_model,
        max_len=512,
        truncation=True,
        padding=True,
    )
    ner = pipeline('ner', grouped_entities=True, model=model_er, tokenizer=tokenizer, )
    clf = pipeline("sentiment-analysis", model=model_rc, tokenizer=tokenizer, )
    qa = pipeline("question-answering", model=model_qa, tokenizer=tokenizer, )
    return ner, clf, qa

def _initialize_input_sources(path_texts: str, path_relation_templates: str, path_entity_aliases: str):
    """
    Loads data required for the method
    Parameters:
    ----------
        path_texts : str
            Path to the TXT file containing cadidate texts
        path_relation_templates : str 
            Path to the TXT file containing the relation templates - one for each relation class 
            (e.g. "какая цена у")
        path_entity_aliases : str 
            Path to the JSON file containing the dictionary of entity class aliases in Russian 
            (e.g. key='Antifreeze', value='антифриз')
    """
    df = _read_and_prepare_data(path_texts) # read texts and split into sentences
    with open(path_entity_aliases) as f:
        lookup = json.load(f)
    with open(path_relation_templates) as f:
        relation_templates = f.readlines()
    return df, relation_templates, lookup

def triplet_extraction(path_texts: str,
    path_relation_templates: str,
    path_entity_aliases: str,
    path_er_model: str,
    path_rc_model: str,
    path_qa_model: str,
    return_unmerged_triples=False, 
):
    """
    Loads data required for the method
    Parameters:
    ----------
        path_texts : str
            Path to the TXT file containing cadidate texts
        path_relation_templates : str 
            Path to the TXT file containing the relation templates - one for each relation class 
            (e.g. "какая цена у")
        path_entity_aliases : str 
            Path to the JSON file containing the dictionary of entity class aliases in Russian 
            (e.g. key='Antifreeze', value='антифриз')
        path_er_model : str
            Path to the pretrained entity recognition model
        path_rc_model : str
            Path to the pretrained relation classification model
        path_qa_model : str
            Path to the pretrained question-answering model
        return_unmerged_triples : bool=False
            Whether to group the found triples by the text source or leave the dataframe with triple as the row
    Returns:
    --------
        df: pandas.core.frame.DataFrame
            Dataframe containing the results of the triplet extraction
    """
    df, relation_templates, lookup = _initialize_input_sources(
        path_texts,
        path_relation_templates,
        path_entity_aliases
    )
    ner, clf, qa = _initialize_pipelines(path_er_model, path_rc_model, path_qa_model)
    # Entity recognition
    # perform ER for subject extraction
    tqdm.pandas(desc="Performing entity recognition...")
    with logging_redirect_tqdm():
        df['ner_res'] = df.progress_apply(lambda x: ner(x.sents), axis=1)
    df = df[df['ner_res'].map(len) > 0] # filter out empty sentences (containing no entities)
    # Process obtained ER results
    df['ners'] = df.ner_res.map(_process_res_ner)
    df['entity_class'] = df.ners.map(lambda x: x.keys())
    df['entity_score'] = df.ners.map(lambda x: list(map(lambda d: max(d, key=lambda v: v['score'])['score'],
                                                x.values())))
    df['entity_string'] = df.ners.map(lambda x: list(map(lambda d: [_['word'] for _ in d], x.values())))
    df['ners'] = df.ners.map(dict.items)
    # Relation classification
    # perform RC to obtain relation
    tqdm.pandas(desc="Performing relation classification...")
    with logging_redirect_tqdm():
        df['rc_res'] = df.progress_apply(lambda x: _classifyer(x.sents, clf), axis=1)
    df = df[df['rc_res'] != -1] # filter out errored by the tokenizer sentences
    # Process obtained RC results
    df['relation_classes'] = df.rc_res.map(lambda x: x[0]['label'])
    df['relation_scores'] = df.rc_res.map(lambda x: x[0]['score'])
    # Explode triples as rows
    df = df.explode(['entity_class', 'entity_score', 'ners', 'entity_string'])
    # Map extracted labels to ids
    df['relation_classes'] = df.relation_classes.map(lambda x: clf.model.config.label2id[x])
    # Filter out empty sentences (containing no relations)
    df = df[df.relation_classes != sorted(pd.unique(df.relation_classes))[-1]]
    # perform QAT for attribute value extraction
    df['question'] = df.apply(lambda x: _create_questions(x, relation_templates, lookup), axis=1)
    tqdm.pandas(desc="Performing question-answering...")
    with logging_redirect_tqdm():
        df['qa_res'] = df.progress_apply(lambda x: qa(x.question, x.text), axis=1)
    df['qa_answer'] = df.qa_res.map(lambda x: x['answer'])
    df['qa_score'] = df.qa_res.map(lambda x: x['score'])
    # calculate triples confidence scores
    df['triplet_confidence'] = df.apply(lambda x: x.entity_score * x.relation_scores * x.qa_score, axis=1)
    if return_unmerged_triples:
        return df
    res = df.reset_index().groupby(['index', 'text'], as_index=False).agg({
        'ners': lambda x: x.tolist(),
        'relation_classes': lambda x: x.tolist(),
        'qa_answer': lambda x: x.tolist(),
        'triplet_confidence': lambda x: x.tolist(),
    })
    res['triples'] = res.apply(lambda x:
        list(zip(x.ners, x.relation_classes, x.qa_answer, x.triplet_confidence)),
        axis=1)
    return res


def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--texts_path",
        type=str,
        required=True,
        help="Path to the triplet containing texts candidates in TXT format",
    )
    parser.add_argument(
        "--relation_templates",
        type=str,
        required=True,
        help="Path to the TXT file containing one relation template for each class ",
    )
    parser.add_argument(
        "--entity_aliases",
        type=str,
        required=True,
        help="Path to the JSON file containing one entity alias for each class {'LABEL' : alias}",
    )
    parser.add_argument(
        "--ner_model_path",
        type=str,
        required=True,
        help="Path to the pretrained model BertForTokenClassification (or model name from Huggingface's hub)",
    )
    parser.add_argument(
        "--rc_model_path",
        type=str,
        required=True,
        help="Path to the pretrained model BertForSequenceClassification (or model name from Huggingface's hub)",
    )
    parser.add_argument(
        "--qat_model_path",
        type=str,
        required=True,
        help="Path to the pretrained model BertForQuestionAnswering (or model name from Huggingface's hub)",
    )
    parser.add_argument(
        "--output_path",
        default='.',
        type=str,
        required=False,
        help="Path to save the results",
    )
    parser.add_argument(
        "--return_unmerged_triples",
        default=False,
        type=bool,
        required=False,
        help="Whether to return the dataframe before grouping the rows by the text value",
    )
    args = parser.parse_args()

    res_df = triplet_extraction(args.texts_path,
        args.relation_templates,
        args.entity_aliases,
        args.ner_model_path,
        args.rc_model_path,
        args.qat_model_path,
        return_unmerged_triples=args.return_unmerged_triples, 
    )
    res_df.to_csv(os.path.join(args.output_path, 'extracted_triples.csv'))
    logger.info(f'Results saved into {os.path.join(args.output_path, "extracted_triples.csv")}')

if __name__=='__main__':
    main()
