import os
import joblib
import numpy.ma as ma
import torch
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer
from dostoevsky.tokenization import RegexTokenizer
from dostoevsky.models import FastTextSocialNetworkModel
import os

from deeppavlov import build_model, configs

import csv
import itertools
import pandas as pd


class SentimentModel:
    def __init__(self, device="cpu"):
        self.model = build_model(configs.classifiers.rusentiment_bert, download=False)

    def predict(self, texts):
        return [self.model([text]) for text in texts]


class RubertModel:
    def __init__(self, model, device="cpu"):
        self.device = device
        self.model = model

    def predict_proba(self, text):
        if isinstance(text, str):
            text = [text]
        print(len(text))
        test_dataloader = self.input_preparation(text)
        logit_preds = []
        for i, batch in enumerate(test_dataloader):
            batch = tuple(t.to(self.device) for t in batch)
            b_input_ids, b_input_mask, b_token_types = batch
            with torch.no_grad():
                outs = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
                b_logit_pred = outs[0]
                pred_label = torch.sigmoid(b_logit_pred)
                b_logit_pred = pred_label.to(self.device).tolist()
            logit_preds += b_logit_pred
        result = np.array(logit_preds)
        return result

    def input_preparation(self, text):
        test_encodings = nn_tokenizer.batch_encode_plus(text, max_length=MAX_LENGTH, padding=True,
                                                        truncation=True)
        test_input_ids = test_encodings['input_ids']
        test_token_type_ids = test_encodings['token_type_ids']
        test_attention_masks = test_encodings['attention_mask']
        test_inputs = torch.tensor(test_input_ids)
        test_masks = torch.tensor(test_attention_masks)
        test_token_types = torch.tensor(test_token_type_ids)
        test_data = TensorDataset(test_inputs, test_masks, test_token_types)
        test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE)
        return test_dataloader

    def predict(self, text, threshold=NN_DEFAULT_THRESHOLD):
        test_dataloader = self.input_preparation(text)
        logit_preds, pred_labels, tokenized_texts = [], [], []
        for i, batch in enumerate(test_dataloader):
            batch = tuple(t.to(self.device) for t in batch)
            b_input_ids, b_input_mask, b_token_types = batch
            with torch.no_grad():
                outs = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
                b_logit_pred = outs[0]
                pred_label = torch.sigmoid(b_logit_pred)
                b_logit_pred = b_logit_pred.detach().cpu().numpy()
                pred_label = pred_label.to('cpu').numpy()
            tokenized_texts.append(b_input_ids)
            logit_preds.append(b_logit_pred)
            pred_labels.append(pred_label)
        pred_labels = [item for sublist in pred_labels for item in sublist]
        pred_bools = [pl > threshold for pl in pred_labels][0]
        return mask_labels(pred_bools)
