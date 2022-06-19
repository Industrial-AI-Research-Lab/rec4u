import pandas as pd
import numpy as np
import transformers
import datasets
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import AutoTokenizer
from typing import NamedTuple, Union, List, Tuple, Sequence


class LoadedDataER(NamedTuple,):
    tokens: Sequence[Sequence[str]]
    labels: Sequence[Sequence[str]]

class EntityRecognitionDataset(Dataset):
    def __init__(self, encodings, labels, tag2id, id2tag):
        self.encodings = encodings
        self.labels = labels
        self.tag2id = tag2id
        self.id2tag = id2tag

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

    @property
    def num_labels(self):
        return len(self.tag2id)

    @property
    def id2label(self):
        return self.id2tag

    @property
    def label2id(self):
        return self.tag2id

def _load_and_process_ner(path: str) -> LoadedDataER:
    """ Loads collected training data from path """
    with open(path, 'r') as f:
        tokens, labels, all_tokens, all_labels = [], [], [], []
        for line in f.readlines():
            strips = line.strip().split()
            if strips:
                tokens.append(strips[0])
                labels.append(strips[1])
            else:
                all_tokens.append(tokens)
                all_labels.append(labels)
                tokens, labels = [], []
        all_tokens.append(tokens)
        all_labels.append(labels)
    return LoadedDataER([_ for _ in all_tokens if _], [_ for _ in all_labels if _])

def _extract_labels(downloaded_data: LoadedDataER) -> List[str]:
    """ Obtained in dataset labels collection """
    _, labels = downloaded_data
    all_labels = [_ for lst in labels for _ in lst]
    return sorted(list(set(all_labels)))

def _tag2id_id2tag(downloaded_data: LoadedDataER) -> Tuple[dict, dict]:
    """ Creates mapping dictionary from literal labels to ids """
    extracted_labels = _extract_labels(downloaded_data)
    tag2id = {name: idx for idx, name in enumerate(extracted_labels)}
    id2tag = {idx: name for idx, name in enumerate(extracted_labels)}
    return tag2id, id2tag

def _tokenize_and_encode_labels(downloaded_data: LoadedDataER,
                               tokenizer: transformers.PreTrainedTokenizerFast,
                               tag2id):
    """ Encodes texts and labels """
    tokens, tags = downloaded_data
    encodings = tokenizer(
            tokens,
            is_split_into_words=True,
            return_offsets_mapping=True,
            padding=True,
            truncation=True
        )
    # Encode BIO tags
    labels = [[tag2id[tag.strip()] for tag in doc] for doc in tags]
    encoded_labels = []
    for doc_labels, doc_offset in zip(labels, encodings.offset_mapping):
        # create an empty array of -100
        doc_enc_labels = np.ones(len(doc_offset),dtype=int) * -100
        arr_offset = np.array(doc_offset)
        # set labels whose first offset position is 0 and the second is not 0
        doc_enc_labels[(arr_offset[:,0] == 0) & (arr_offset[:,1] != 0)] = doc_labels
        encoded_labels.append(doc_enc_labels.tolist())

    encodings.pop("offset_mapping") # we don't want to pass this to the model
    return encodings, encoded_labels

def prepare_ner_data(path, tokenizer: transformers.PreTrainedTokenizerFast):
    """
    Creates instance of Dataset class
    Parameters:
    ----------
        path : str
        tokenizer : transformers.PreTrainedTokenizerFast
    Returns:
    -------
        dataset : EntityRecognitionDataset
    """
    downloaded_data = _load_and_process_ner(path)
    tag2id, id2tag = _tag2id_id2tag(downloaded_data)
    encodings, encoded_labels = _tokenize_and_encode_labels(downloaded_data, tokenizer, tag2id)
    dataset = EntityRecognitionDataset(encodings, encoded_labels, tag2id, id2tag)
    return dataset
