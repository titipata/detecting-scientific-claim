"""
Example script to predict claim using technique from the following paper
``A Simple but Tough-to-Beat Baseline for Sentence Embeddings`` combining 
with discourse probability
"""
from typing import List
import os
import json
import sys
import numpy as np
import pandas as pd
from nltk import word_tokenize
import torch.nn.functional as F
import torch
from fastText import load_model
sys.path.insert(0, '..')

from allennlp.common.file_utils import cached_path
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor
from allennlp.modules.token_embedders.embedding import EmbeddingsTextFile
from allennlp.data.instance import Instance
from allennlp.data.fields import TextField, SequenceLabelField, ListField
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer

from discourse import read_json
from discourse.predictors.discourse_predictor import DiscourseClassifierPredictor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.decomposition import TruncatedSVD

EMBEDDING_DIM = 200
MEDLINE_WORD_PATH = 'https://s3-us-west-2.amazonaws.com/pubmed-rct/medline_word_prob.json'
DISCOURSE_MODEL_PATH = 'https://s3-us-west-2.amazonaws.com/pubmed-rct/model.tar.gz'
PUBMED_PRETRAINED_PATH = 'https://s3-us-west-2.amazonaws.com/pubmed-rct/wikipedia-pubmed-and-PMC-w2v.txt.gz'
TRAIN_PATH = 'https://s3-us-west-2.amazonaws.com/pubmed-rct/train_labels.json'
VALIDATION_PATH = 'https://s3-us-west-2.amazonaws.com/pubmed-rct/validation_labels.json'
TEST_PATH = 'https://s3-us-west-2.amazonaws.com/pubmed-rct/test_labels.json'

archive = load_archive(DISCOURSE_MODEL_PATH) # discourse model
predictor = Predictor.from_archive(archive, 'discourse_predictor')
assert os.path.exists('wiki.en.bin') == True
ft_model = load_model('wiki.en.bin') # fastText word vector
p_dict = json.load(open(cached_path(MEDLINE_WORD_PATH), 'r'))


def read_embedding(pretrained_path=PUBMED_PRETRAINED_PATH):
    """
    Read Pubmed Pretrained embedding from Amazon S3 and 
    return dictionary of embeddings
    """
    embeddings = {}
    with EmbeddingsTextFile(pretrained_path) as embeddings_file:
        for line in embeddings_file:
            token = line.split(' ', 1)[0]
            if token in p_dict.keys():
                fields = line.rstrip().split(' ')
                vector = np.asarray(fields[1:], dtype='float32')
                embeddings[token] = vector
    return embeddings


def get_sentence_vector(sent, ft_model, a=10e-4):
    """Average word vector for given list of words using fastText
    ref: A Simple but Tough-to-Beat Baseline for Sentence Embeddings, ICLR 2017
    https://openreview.net/pdf?id=SyK00v5xx
    """
    words = word_tokenize(sent)
    wvs = []
    for w in words:
        wv = ft_model.get_word_vector(w, None)
        if w is not None and wv is not None:
            wvs.append([wv, p_dict.get(w, 0)])
    wvs = np.array(wvs)
    return np.mean(wvs[:, 0] * (a / (a + wvs[:, 1])), axis=0)


def text_to_instance(sents: List[str], labels: List[str] = None):
    """
    Make list of sentences (and labels) to ``Instance``
    """
    fields = {}
    tokenized_sents = [WordTokenizer().tokenize(sent) for sent in sents]
    sentence_sequence = ListField([TextField(tk, {'tokens': SingleIdTokenIndexer()}) for tk in tokenized_sents])
    fields['sentences'] = sentence_sequence
    if labels is not None:
        fields['labels'] = SequenceLabelField(labels, sentence_sequence)
    return Instance(fields)


def calculate_pc(X, n_comp=1):
    """
    Calculate first principal component from the given matrix X
    """
    svd = TruncatedSVD(n_components=n_comp, n_iter=100, random_state=0)
    svd.fit(X)
    return svd.components_


def remove_pc(x, pc):
    """
    Remove first principal component from an array
    """
    return x - x.dot(pc.transpose()) * pc


def flatten_dataset(df):
    """
    Flatten Gold Standard JSON data for Claim Extraction
    """
    sentence_data = []
    for _, r in df.iterrows():
        sentence_data.extend(list(zip(r.sentences, r.labels)))
    flatten_df = pd.DataFrame(sentence_data, columns=['sentence', 'label'])
    return flatten_df


if __name__ == '__main__':
    # read dataset
    train_df = pd.DataFrame(read_json(cached_path(TRAIN_PATH)))
    validation_df = pd.DataFrame(read_json(cached_path(VALIDATION_PATH)))
    test_df = pd.DataFrame(read_json(cached_path(TEST_PATH)))

    train_df = flatten_dataset(train_df)
    validation_df = flatten_dataset(validation_df)
    test_df = flatten_dataset(test_df)

    # prepare dataset
    train_df['class_probability'] = train_df.sentence.map(lambda x: predictor.predict_json({'sentence': x})['class_probabilities'])
    train_df['sentence_vector'] = train_df.sentence.map(lambda x: get_sentence_vector(x, ft_model))
    X_train = np.hstack((np.vstack(train_df['sentence_vector'].values), np.vstack(train_df['class_probability'].values)))
    y_train = train_df.label.astype(int).values

    validation_df['class_probability'] = validation_df.sentence.map(lambda x: predictor.predict_json({'sentence': x})['class_probabilities'])
    validation_df['sentence_vector'] = validation_df.sentence.map(get_sentence_vector)
    X_val = np.hstack((np.vstack(validation_df['sentence_vector'].values), 
                       np.vstack(validation_df['class_probability'].values)))
    y_val = validation_df.label.astype(int).values

    test_df['class_probability'] = test_df.sentence.map(lambda x: predictor.predict_json({'sentence': x})['class_probabilities'])
    test_df['sentence_vector'] = test_df.sentence.map(lambda x: get_sentence_vector(x, ft_model))
    X_test = np.hstack((np.vstack(test_df['sentence_vector'].values), 
                        np.vstack(test_df['class_probability'].values)))
    y_test = test_df.label.astype(int).values

    logist_model = LogisticRegression()
    logist_model.fit(X_train, y_train)

    # print optimal threshold for validation dataset
    p_claim = logist_model.predict_proba(X_val)
    for threshold in np.arange(0.25, 0.7, 0.02):
        y_pred = (p_claim[:, 1] >= threshold)
        print('Threshold = {}, F1-score = {}'.format(threshold, f1_score(y_val, y_pred)))

    # precision, recall, f-score for validation dataset
    y_pred = logist_model.predict_proba(X_val)
    y_true = y_val
    y_pred = (y_pred[:, 1] >= 0.49).astype(int)
    print(precision_recall_fscore_support(y_true, y_pred, average='binary'))