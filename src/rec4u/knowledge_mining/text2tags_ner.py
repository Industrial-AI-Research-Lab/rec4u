import logging
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from conllu import parse
import conllu
import json
from multiprocessing import Pool #
from razdel import sentenize
import argparse
import pickle
import os
import pymorphy2
from pymorphy2.tokenizers import simple_word_tokenize
import re

morph = pymorphy2.MorphAnalyzer()
logger = logging.getLogger(__name__)

def _normalize_text(sentence: str, return_lemmas_list: bool=False, digits_names_left: bool=False):
    """
    Parameters:
    ----------
        sentence : str
            String to normalize
        return_lemmas_list : bool = False
            Whether to return normalized text or a list of normalized text's tokens
        digits_names_left : bool = False
            Whether to keep digit and latin tokens untouched
    Returns:
    -------
        Lemmatized text version
    """
    if digits_names_left:
        lemmas = [morph.parse(i)[0].normal_form if not re.search('[\dA-z]', i) else i for i in simple_word_tokenize(sentence)]
    else:
        lemmas = [morph.parse(i)[0].normal_form for i in simple_word_tokenize(sentence)]
    if return_lemmas_list:
        return lemmas
    return " ".join(lemmas)

def _labels(normalized_sentence, terms_dictionary):
    """ Finds longest substring of NERs presented in a sentence given a {term: label} dictionary """
    normalized_dict = list(map(lambda x: x.split(), terms_dictionary.keys()))
    found_terms = {v: [] for v in terms_dictionary.values()}
    for item in normalized_dict:
        if not set(item) - set(normalized_sentence):
            found_terms[terms_dictionary[" ".join(item)]].append((" ".join(item), len(item)))
    for k, v in found_terms.items():
        if v:
            _, count = max(v, key=lambda item: item[1])
            found_terms[k] = set([term for term, c in v if c == count])
    return {k: ' '.join(v) for k, v in found_terms.items() if v}

def _labeler_er(normalized_sentence, out_labels):
    """ Creates IOB labels from normalized sentence and `labels` func output """
    res = []
    suffix = {key: 'B-' for key in out_labels.keys()}
    prev_positions = {key: 100_000 for key in out_labels.keys()}
    for idx, token in enumerate(normalized_sentence):
        lbl = 'O'
        for key, value in out_labels.items():
            if token in value.split() and idx - prev_positions[key] < 6:
                lbl = suffix[key] + key
                prev_positions[key] = idx
                k = key
        res.append(lbl)
        if lbl != 'O':
            suffix[k] = 'I-'
    return res

def _process_text(text):
    """ Processes text into lemmatized and tokenized sentences """
    return [_normalize_text(_.text, return_lemmas_list=True) for _ in sentenize(text)]

def _terms_search(normalized_sentence, terms_dictionary):
    """ Returns flag True if the sentence is the candidate for ER extraction """
    set_of_terms = set([_ for lst in list(map(lambda x: x.split(),
                        terms_dictionary.keys())) for _ in lst])
    if set_of_terms.intersection(set(normalized_sentence)):
        return True

def label_text(text: str, terms_dictionary: dict):
    """
    Creates IOB labels from text
    Parameters:
    ----------
        text : str
            Text to find ERs
        terms_dictionary : dict
            Dictionary with terms as keys and labels as values
    Returns:
    -------
        tokenized_texts : List[str]
            List with text tokens
        text_labels : List[str]
            List with IOB labels
    """
    text_labels = []
    normalized_text = _process_text(text)
    termed_sent = list(map(lambda x: _terms_search(x, terms_dictionary), normalized_text))
    for idx, sentence in enumerate(normalized_text):
        if not termed_sent[idx]:
            text_labels += ['O'] * len(sentence)
        else:
            text_labels += _labeler_er(sentence, _labels(sentence, terms_dictionary))
    return simple_word_tokenize(text), text_labels

def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dict_path",
        type=str,
        required=True,
        help="Path to the {term: label} dictionary in JSON format"
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
        help="Path to save the tokenized text and output labels in TXT format"
    )

    args = parser.parse_args()

    assert args.dict_path.endswith('.json'), 'Dictionary with class names should be in JSON format'
    assert args.texts_path.endswith('.txt'), 'Texts file should be in TXT format'


    with open(args.dict_path) as f:
        lookup_dict = json.load(f)

    with open(args.texts_path) as f:
        texts = f.read()
    texts = texts.split('\n')

    labels_ = []
    tokenized_texts_ = []
    with logging_redirect_tqdm():
        for y in tqdm(texts, desc='Processing text files'):
            tokenized_text, text_labels = label_text(y, lookup_dict)
            labels_.append(text_labels)
            tokenized_texts_.append(tokenized_text)

    with open(os.path.join(args.output_dir, 'training_data_er.txt'), 'w') as f:
        for text_idx, tags in enumerate(labels_):
            f.write('\n'.join([' '.join([tokenized_texts_[text_idx][token_idx], label])
                               for token_idx, label in enumerate(tags)]) + '\n\n')


if __name__ == '__main__':
    main()
