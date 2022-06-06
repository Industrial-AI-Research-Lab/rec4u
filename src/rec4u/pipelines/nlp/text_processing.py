import re
import pymystem3
from nltk import ngrams

r_num = re.compile(r'(^[0-9]+)')
r_punct = re.compile(r'[."\[\]/,()!?;:*#|\\%^$&{}~_`=\-@•]')
r_white_space = re.compile(r'\s{2,}')
r_html = re.compile(r'(\<[^>]*\>)')
r_words = re.compile(r'\W+')
r_date = re.compile(r'\d+.\d+.\d+')
r_site = re.compile(r'(http[s]?://|www)(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')

m = pymystem3.Mystem()

def process_text(text):
    try:
        text = r_html.sub(" ", text)
    except:
        return ''
    text = text.lower()
    text = r_site.sub(" САЙТ ", text)
    text = r_date.sub(" ДАТА ", text)
    text = r_punct.sub(" ", text)
    text = r_num.sub(" ", text)
    text = re.sub('\uf0fc', ' ', text)
    text = r_white_space.sub(" ", text)
    text = ' '.join(text.split())
    return text.strip()

def process_req(list_of_req):
    all_lemmatized = []
    for req_i in list_of_req:
        list_j = []
        try:
            for req_j in req_i.split(' || '):
                processed = process_text(req_j)
                list_j.append(processed)
            all_text = ' || '.join(list_j)
            lemmatized = ''.join(m.lemmatize(all_text)).strip()
            all_lemmatized.append(lemmatized)
        except:
            all_lemmatized.append('')
    return all_lemmatized

def get_bigrams(text):
    return create_ngrams(text, 2)


def get_trigrams(text):
    return create_ngrams(text, 3)


def lemmatize_data(text):
    return ''.join(m.lemmatize(text)).strip()


def create_ngrams(req, n=2):
    list_of_terms = []
    terms = ngrams(req.split(), n)
    for gram in terms:
        list_of_terms.append('_'.join(gram))
    return ' || '.join(list_of_terms)