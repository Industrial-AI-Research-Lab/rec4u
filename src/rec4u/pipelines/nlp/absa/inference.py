import math
import os
import sys
import random
import json
import argparse
import re
from collections import namedtuple, Counter, defaultdict

from bertopic import BERTopic
from cleantext import clean

import numpy as np
import pandas as pd

from transformers import BertModel

from sklearn import metrics 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

sys.path.append('ABSA-PyTorch')

from data_utils import build_tokenizer, build_embedding_matrix, Tokenizer4Bert, ABSADataset, Dataset

from models.aen import CrossEntropyLoss_LSR, AEN_BERT
from models.bert_spc import BERT_SPC
from data_utils import pad_and_truncate

import pymorphy2

class ABSADatasetCode(Dataset):
    def __init__(self, lines, tokenizer):
        
        all_data = []
        for i in range(0, len(lines), 3):
            text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
            aspect = lines[i + 1].lower().strip()
            polarity = lines[i + 2].strip()

            text_raw_indices = tokenizer.text_to_sequence(text_left + " " + aspect + " " + text_right)
            text_raw_without_aspect_indices = tokenizer.text_to_sequence(text_left + " " + text_right)
            text_left_indices = tokenizer.text_to_sequence(text_left)
            text_left_with_aspect_indices = tokenizer.text_to_sequence(text_left + " " + aspect)
            text_right_indices = tokenizer.text_to_sequence(text_right, reverse=True)
            text_right_with_aspect_indices = tokenizer.text_to_sequence(" " + aspect + " " + text_right, reverse=True)
            aspect_indices = tokenizer.text_to_sequence(aspect)
            left_context_len = np.sum(text_left_indices != 0)
            aspect_len = np.sum(aspect_indices != 0)
            aspect_in_text = torch.tensor([left_context_len.item(), (left_context_len + aspect_len - 1).item()])
            polarity = int(polarity) + 1

            text_bert_indices = tokenizer.text_to_sequence('[CLS] ' + text_left + " " + aspect + " " + text_right + ' [SEP] ' + aspect + " [SEP]")
            bert_segments_ids = np.asarray([0] * (np.sum(text_raw_indices != 0) + 2) + [1] * (aspect_len + 1))
            bert_segments_ids = pad_and_truncate(bert_segments_ids, tokenizer.max_seq_len)

            text_raw_bert_indices = tokenizer.text_to_sequence("[CLS] " + text_left + " " + aspect + " " + text_right + " [SEP]")
            aspect_bert_indices = tokenizer.text_to_sequence("[CLS] " + aspect + " [SEP]")

            data = {
                'text_bert_indices': text_bert_indices,
                'bert_segments_ids': bert_segments_ids,
                'text_raw_bert_indices': text_raw_bert_indices,
                'aspect_bert_indices': aspect_bert_indices,
                'text_raw_indices': text_raw_indices,
                'text_raw_without_aspect_indices': text_raw_without_aspect_indices,
                'text_left_indices': text_left_indices,
                'text_left_with_aspect_indices': text_left_with_aspect_indices,
                'text_right_indices': text_right_indices,
                'text_right_with_aspect_indices': text_right_with_aspect_indices,
                'aspect_indices': aspect_indices,
                'aspect_in_text': aspect_in_text,
                'polarity': polarity,
            }

            all_data.append(data)
        self.data = all_data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
    
def data2predict_only(tokenizer, sample):
    text = sample['sentence']
    
    _label = sample['label']
    _text = text.replace(_label, '$T$')
    text_left, _, text_right = [s.lower().strip() for s in _text.partition("$T$")]

    text_raw_bert_indices = torch.tensor(tokenizer.text_to_sequence("[CLS] " + text_left + " " + _label + " " + text_right + " [SEP]")).unsqueeze(0)
    aspect_bert_indices = torch.tensor(tokenizer.text_to_sequence("[CLS] " + _label + " [SEP]")).unsqueeze(0)
    
    return {'text_bert': text_raw_bert_indices, 
                  'aspect_bert': aspect_bert_indices,
                  'text': _text,
                  'aspect': _label}

_opt = namedtuple('opt', ['max_seq_len', 
                          'pretrained_bert_name', 
                          'device', 
                          'best_model_path', 
                          'model_class', 
                          'dropout',
                          'bert_dim', 
                          'hidden_dim', 
                          'polarities_dim', 
                          'hops', 
                          'batch_size'])

class Infurrer:
    def __init__(self, opt):
        self.opt = opt

        tokenizer = Tokenizer4Bert(opt.max_seq_len, opt.pretrained_bert_name)
        bert = BertModel.from_pretrained(opt.pretrained_bert_name)
        self.model = opt.model_class(bert, opt).to(opt.device)
        
        self.model.load_state_dict(torch.load(opt.best_model_path))
        self.model.eval()

    def infer(self, sample):
        with torch.no_grad():
            outputs = self.model(sample)
            
        return outputs

def parse_jsonline(data):
    d = defaultdict(list)

    if data['reviews'] is None or data['reviews']['data'] is None:
        return None

    for review_data in data['reviews']['data']:
        text = [review_data['body'], review_data['pros'], review_data['cons']]

        for i in range(len(text)):
            t = text[i]
            if t is None:
                text[i] = ''
            elif t[-1] not in list('.!?'):
                text[i] = t + '.'

        rev = f'{text[0]} {text[1]} {text[2]}'

        d['articul'].append(data['articul'])
        d['name'].append(data['displayedName'])
        d['brand'].append(data['brand'])
        d['review'].append(rev)

    return d

def get_items_topics(reviews_df, topic_probs, topic2theme, general_themes):
    items_topics = {} 

    articuls = reviews_df.articul.unique()

    for articul in articuls:
        item_reviews = reviews_df[reviews_df.articul == articul]
        item_topics = []
        for idx in item_reviews.index:
            most_prob_topics = topic_probs[idx].argsort()[-2:][::-1]
            topic = most_prob_topics[0]
            for t in most_prob_topics:
                if t != -1:
                    topic = t
                    break
            item_topics.append(topic)
        c = Counter(item_topics)
        item_topics = []
        for k, v in c.items():
            if (v >= 0.1 * len(item_reviews)) and (k in topic2theme.keys()):
                item_topics.append(topic2theme[k])

        item_topics.extend(general_themes.keys())
        items_topics[int(articul)] = item_topics
    
    return items_topics

def remove_emojis(data):
    emoj = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", re.UNICODE)
    return re.sub(emoj, '', data)

def apply_clean(doc):
    doc = clean(doc,
                fix_unicode=True,               # fix various unicode errors
                to_ascii=False,                  # transliterate to closest ASCII representation
                lower=True,                     # lowercase text
                no_line_breaks=True,           # fully strip line breaks as opposed to only normalizing them
                no_urls=True,                  # replace all URLs with a special token
                no_emails=True,                # replace all email addresses with a special token
                no_phone_numbers=False,         # replace all phone numbers with a special token
                no_numbers=False,               # replace all numbers with a special token
                no_digits=False,                # replace all digits with a special token
                no_currency_symbols=False,      # replace all currency symbols with a special token
                no_punct=False,                 # remove punctuations
                replace_with_punct="",          # instead of removing punctuations you may replace them
                replace_with_url="<URL>",
                replace_with_email="<EMAIL>",
                lang="en"                       # set to 'de' for German special handling
               )
    doc = remove_emojis(doc)
    doc = re.sub(r"""[\!\?\.]""", ' . ', doc)
    doc = re.sub(r"""['"\(\)\-%\+\*:;,]""", ' ', doc)
    return doc

def get_reviews_df(path):
    part_dfs = []

    for filename in os.listdir(path):
        with open(os.path.join(path, filename), 'r') as f:
            lines = f.read().splitlines()

        for line in lines:

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            parsed = parse_jsonline(data)
            if parsed is not None:
                part = pd.DataFrame.from_dict(parsed)
                part_dfs.append(part)

    reviews_df = pd.concat(part_dfs).reset_index(drop=True)
    reviews_df['review_cleaned'] = reviews_df.review.apply(apply_clean)
    return reviews_df

regex = re.compile('[^a-zA-Z]')
morph = pymorphy2.MorphAnalyzer()

def get_predictions(model, tokenizer, reviews_df, aaaa_processed, items_topics):
    predictions = []

    for i, row in enumerate(reviews_df.itertuples(index=False)):
        if i % 10000 == 0:
            print(i)

    #     aaaa_topics_keys = set(aaaa_processed.keys()) & set(row.topics)

        aaaa_topics_processed = {topic: aaaa_processed[topic] for topic in items_topics[row.articul]}

        sentences = str(row.review_cleaned)
        sentences = sentences.replace('. ', '\n').replace(':', '\n').replace('! ', '\n')
        sentences = sentences.split('\n')
        for sentence in sentences[1:]:
            sentence = sentence.lower()
            _tokens = texts2tokens(sentence)
            tokens = list(_tokens.keys())
            uniq_tokens = set(tokens)
            if len(tokens) > 2:
                words = list(_tokens.values())

            for key, values in aaaa_topics_processed.items():
                general_tokens = (set(values) & uniq_tokens)
                if len(general_tokens) > 0:
                    for target in general_tokens:
                        _sample = data2predict_only(tokenizer, {'sentence': sentence.replace(_tokens[target], '$T$'),
                           'label': _tokens[target]})

                        sample = (_sample['text_bert'].to('cuda:0'), _sample['aspect_bert'].to('cuda:0'))
                        predict = model.infer(sample).cpu().detach().numpy()[0]
                        text = sentence
                        aspect = target
                        sentiment = predict.argmax(-1) - 1
                        power = predict

                        predictions.append({'articul': row.articul, 'name': row.name, 'brand': row.brand, 'text': text, 'sentiment': sentiment, 'topic': key, 'aspect_term': target, 'power': power,})
    return predictions

def texts2tokens(text):
    text = text.lower()
    parsed_words = (morph.parse(''.join([i for i in w if i.isalpha()])) for w in text.split(' '))
    analyze = {'{}_{}'.format(i[0].normal_form, i[0].tag.POS): i[0].word for i in parsed_words}
    return analyze

def get_items_top_aspects(predictions):
    predictions_pdf = pd.DataFrame(predictions)
    predictions_pdf = predictions_pdf.drop_duplicates(['text', 'aspect_term'])
    predictions_pdf['power_'] = predictions_pdf.power.apply(lambda x: min([abs(i - max(x)) for i in x if abs(i - max(x)) != 0]))
    predictions_pdf = predictions_pdf[(predictions_pdf.power_ > 0.5) & (predictions_pdf.sentiment != 0)]
    predictions_pdf_grouped = predictions_pdf.groupby(['aspect_term', 'articul', 'name']).agg(
        {'sentiment': list}
    ).reset_index()
    predictions_pdf_grouped['pos'] = predictions_pdf_grouped.sentiment.apply(lambda x: len([i for i in x if i > 0]))
    predictions_pdf_grouped['neg'] = predictions_pdf_grouped.sentiment.apply(lambda x: len([i for i in x if i < 0]))
    predictions_pdf_grouped['neu'] = predictions_pdf_grouped.sentiment.apply(lambda x: len([i for i in x if i == 0]))
    
    positive_predictions = predictions_pdf_grouped.copy()
    positive_predictions['pos_ratio'] = positive_predictions.pos / (positive_predictions.neg + 1)
    positive_predictions['num'] = positive_predictions[positive_predictions.pos_ratio > 1.0].sort_values('pos_ratio', ascending=False).groupby('articul').cumcount()
    positive_predictions = positive_predictions[positive_predictions.num < 3.0]
    
    negative_predictions = predictions_pdf_grouped.copy()
    negative_predictions['neg_ratio'] = negative_predictions.neg / (negative_predictions.pos + 1)
    negative_predictions['num'] = negative_predictions[negative_predictions.neg_ratio > 1.0].sort_values('neg_ratio', ascending=False).groupby('articul').cumcount()
    negative_predictions = negative_predictions[negative_predictions.num < 3.0]
    
    items_df = predictions_pdf[['articul', 'name', 'brand']].drop_duplicates()
    
    for i in range(3):
        items_df = items_df\
        .merge(positive_predictions.loc[positive_predictions.num.astype(int) == i, ['articul', 'aspect_term', 'pos']]\
               .rename(columns={'aspect_term': f'pos_aspect_{i+1}', 'pos': f'pos_aspect_count_{i+1}'}), on='articul', how='left')

    for i in range(3):
        items_df = items_df\
        .merge(negative_predictions.loc[negative_predictions.num.astype(int) == i, ['articul', 'aspect_term', 'neg']]\
               .rename(columns={'aspect_term': f'neg_aspect_{i+1}', 'neg': f'neg_aspect_count_{i+1}'}), on='articul', how='left')
        
    return items_df
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('reviews_path', help='path to folder with jsonlines files with reviews', type=str)
    parser.add_argument('--bertopic_model', default='bertopic.model', help='path to builded bertopic model', type=str)
    parser.add_argument('--topic2theme', default='topic2theme.json', help='path to json file with topic to theme mapping', type=str)
    parser.add_argument('--general_aspects', default='general_aspects.json', help='path to json file with general aspects', type=str)
    parser.add_argument('--aspects_processed', default='aspects_processed.json', help='path to json file with all processed aspects', type=str)
    parser.add_argument('--bert_name', default='sberbank-ai/ruBert-base', help='name of bert from hugginface repository', type=str)
    parser.add_argument('--state_dict', default='state_dict/aen_bert_auto2_val_acc_0.7786', help='path to state dict of trained aen bert', type=str)
    parser.add_argument('--output_path', default='items_top_3_aspects_example.json', help='path to output json file', type=str)
    opt = parser.parse_args()
    
    reviews_df = get_reviews_df(opt.reviews_path)
    
    topic_model = BERTopic.load(opt.bertopic_model)
    _, probs = topic_model.transform(reviews_df.review_cleaned.tolist())
    
    with open(opt.topic2theme) as f:
        topic2theme = json.load(f)
        
    with open(opt.general_aspects) as f:
        general_themes = json.load(f)
        
    items_topics = get_items_topics(reviews_df, probs, topic2theme, general_themes)
    
    with open(opt.aspects_processed) as f:
        aaaa_processed = json.load(f)
    
    bert_opt = _opt(80, opt.bert_name, 'cuda:0', opt.state_dict, AEN_BERT, 0.1, 768, 300, 3, 3, 1)
    tokenizer = Tokenizer4Bert(100, opt.bert_name)
    model = Infurrer(bert_opt)
    
    predictions = get_predictions(model, tokenizer, reviews_df, aaaa_processed, items_topics)
    
    items_df = get_items_top_aspects(predictions)    
    items_df.to_json(opt.output_path)    

if __name__ == '__main__':
    main()