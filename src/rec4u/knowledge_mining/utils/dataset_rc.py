import pandas as pd
import numpy as np
import transformers
import datasets
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import AutoTokenizer
from typing import NamedTuple, Union, List, Tuple, Sequence

class LoadedDataRC(NamedTuple,):
    texts: Sequence[str]
    labels: Sequence[Union[int, str]]

class RelationClassificationDataset(Dataset):
    def __init__(self, encoded_inputs, labels, tag2id=None, id2tag=None):
        self.attention_mask = encoded_inputs['attention_mask']
        self.input_ids = encoded_inputs['input_ids']
        self.token_type_ids = encoded_inputs['token_type_ids']
        self.labels = labels
        self.tag2id = tag2id
        self.id2tag = id2tag

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {
            'attention_mask': self.attention_mask[index],
            'input_ids':  self.input_ids[index],
            'token_type_ids': self.token_type_ids[index],
            'labels' : self.labels[index]
        }

    @property
    def num_labels(self):
        return len(set(self.labels))

    @property
    def id2label(self):
        return self.id2tag

    @property
    def label2id(self):
        return self.tag2id


def _load_and_process_rc(path: str) -> LoadedDataRC:
    """Loads collected training data from path """
    df = pd.read_csv(path, index_col=0)
    texts = df['text'].tolist()
    labels = df['class'].tolist()
    return LoadedDataRC(texts, labels)


def _extract_labels(downloaded_data: LoadedDataRC) -> List[Union[str, int]]:
    """ Obtained in dataset labels collection """
    _, labels = downloaded_data
    return sorted(list(set(labels)))


def _tag2id_id2tag(downloaded_data: LoadedDataRC) -> Tuple[dict, dict]:
    """ Creates mapping dictionary from literal labels to ids """
    extracted_labels = _extract_labels(downloaded_data)
    tag2id = {name: idx for idx, name in enumerate(extracted_labels)}
    id2tag = {idx: name for idx, name in enumerate(extracted_labels)}
    return tag2id, id2tag

def _tokenize_texts_and_encode_tags(
    downloaded_data: LoadedDataRC,
    tokenizer: transformers.PreTrainedTokenizerFast,
):
    """ Prepares dataset and labels for training """
    texts, labels = downloaded_data
    encoded_inputs = tokenizer(texts, padding='max_length', truncation=True, return_tensors='pt')
    if not all([isinstance(_, int) for _ in labels]):
        tag2id, id2tag = _tag2id_id2tag(downloaded_data)
        labels = [tag2id[_.strip()] for _ in labels]
        return encoded_inputs, labels, tag2id, id2tag
    return encoded_inputs, labels

def prepare_rc_data(path, tokenizer: transformers.PreTrainedTokenizerFast):
    """
    Creates instance of Dataset class
    Parameters:
    ----------
        path : str
        tokenizer : transformers.PreTrainedTokenizerFast
    Returns:
    -------
        dataset : RelationClassificationDataset
    """
    downloaded_data = _load_and_process_rc(path)
    encoded_data = _tokenize_texts_and_encode_tags(downloaded_data, tokenizer)
    dataset = RelationClassificationDataset(*encoded_data)
    return dataset
