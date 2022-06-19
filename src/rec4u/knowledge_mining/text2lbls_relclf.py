from sentence_transformers import SentenceTransformer, util
import json
import numpy as np
import torch
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
import logging
import argparse
import os
from razdel import sentenize

logger = logging.getLogger(__name__)

def _sentenize(text_:str):
    return [_.text for _ in sentenize(text_)]

def _get_scores(model, relation_examples, texts):
    """ Embeds the texts and example relations to obtain scores and candidate labels """
    encoded_examples = model.encode(relation_examples)
    encoded_texts = model.encode(texts)
    sts_scores = util.pytorch_cos_sim(encoded_examples, encoded_texts)
    vals, idx = sts_scores.max(dim=0)
    return vals, idx

def _get_labels(texts, vals, idx, top_n=200, range_threshold=.3, thresholds=[]):
    """ Returns two dictionaries, containing labeled texts and similarity scores {class_label: texts}, {class_label: scores} """
    texts = np.array(texts)
    labels_dict, scores_dict = {}, {}
    taken_ids = []
    for i, lbl in enumerate(torch.unique(idx).tolist()):
        class_labels = vals[torch.where(idx == lbl)[0]] # Extract class values
        # Define threshold for scores
        if thresholds:
            assert len(thresholds) == len(torch.unique(idx)), \
            'Argument `thresholds` should provide a threshold value for each class'
            class_threshold = thresholds[i]
        else:
            class_threshold = min(class_labels) + range_threshold * (max(class_labels) - min(class_labels))
        # Texts satisfying the conditions on the filtering
        class_idx = torch.where((vals >= class_threshold) & (idx == lbl))[0]
        sorted_class_scores, sorted_class_idx = vals[class_idx].sort(descending=True)
        # Top n texts in class
        # Top n texts in class
        sorted_texts = texts[class_idx[sorted_class_idx]]
        if isinstance(sorted_texts, str):
            class_texts = np.array([sorted_texts])[:top_n]
        else:
            class_texts = sorted_texts[:top_n]
        labels_dict[lbl] = class_texts
        scores_dict[lbl] = sorted_class_scores[:top_n]
        taken_ids += class_idx[sorted_class_idx][:top_n].tolist()
    # Gather other texts into negative examples class
    lbls_no_class, idxs_no_class = vals.sort(descending=False)
    if top_n:
        labels_dict[i + 1] = texts[idxs_no_class[:top_n]]
        scores_dict[i + 1] = lbls_no_class[:top_n]
    else:
        labels_dict[i + 1] = texts[list(set(idxs_no_class.tolist()) - set(taken_ids))]
        scores_dict[i + 1] = lbls_no_class[list(set(idxs_no_class.tolist()) - set(taken_ids))]
    return labels_dict, scores_dict

def label_texts(model, texts, relation_examples, **kwargs):
    """
    Creates pandas dataframe with columns: `text`, `class`, `score`
    Parameters:
    ----------
        model : sentence_transformers.SentenceTransformer.SentenceTransformer
            Sentence transformer model for text embeddings
        texts : list[str]
            List of texts
        relation_examples : list[str]
            List of relation examples (one example for each class)
    Returns:
    -------
        dataframe : pandas.core.frame.DataFrame
            Dataframe with labels and scores obtained
    """
    vals, idx = _get_scores(model, relation_examples, texts)
    texts, labels = _get_labels(texts, vals, idx,
        top_n=kwargs['top_n'],
        range_threshold=kwargs['range_threshold'],
        thresholds=kwargs['thresholds'])
    res = list(zip(*[(lbl, sent) for lbl, text in texts.items() for sent in text]))
    df = pd.DataFrame({
             'text': res[1],
             'class': res[0],
             'score': [score.item() for lbl, label in labels.items() for score in label],
         })
    return df

def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--relation_examples",
        type=str,
        required=True,
        help="Path to the TXT file containing one relation example for each class"
    )
    parser.add_argument(
        "--texts_path",
        type=str,
        required=True,
        help="Path to the texts file in TXT format"
    )
    parser.add_argument(
        "--output_dir",
        default='.',
        type=str,
        required=False,
        help="Path to save the output labels in pickle format"
    )
    parser.add_argument(
        "--top_n",
        default=200,
        type=int,
        required=False,
        help="Number of top relation examples for each class"
    )
    parser.add_argument(
        "--range_threshold",
        default=0.2,
        type=float,
        required=False,
        help="Relative threshold for all the relation classes to create examples. \
        E.g. If the minimum score in class candidates is 0.3 and maximum is 0.6, \
        with the threshold of 0.4 all candidates with score larger or equal 0.42 \
        will be chosen as positive class examples"
    )
    parser.add_argument(
        "--thresholds",
        default='',
        type=str,
        required=False,
        help="Absolute threshold for all the classes, the input should be given in \
        a form of a string, e.g. `0.34,0.65,0.24` for 3 relation classes"
    )
    args = parser.parse_args()

    assert args.texts_path.endswith('.txt'), 'Texts file should be in TXT format'
    assert args.relation_examples.endswith('.txt'), 'Relation examples should be in TXT format'

    with open(args.texts_path) as f:
        texts = f.readlines()
    texts_ = []
    for text in texts:
        texts_ += _sentenize(text)

    with open(args.relation_examples) as f:
        relation_examples = f.readlines()

    model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
    thresholds = args.thresholds.split(',') if args.thresholds else []
    df = label_texts(model,
        texts_,
        relation_examples,
        top_n=args.top_n,
        range_threshold=args.range_threshold,
        thresholds=thresholds,
    )

    df.to_csv(os.path.join(args.output_dir, 'training_data_rc.csv'))

if __name__ == '__main__':
    main()
