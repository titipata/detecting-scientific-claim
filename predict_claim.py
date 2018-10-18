from typing import List
import os
import json
import numpy as np
from nltk import word_tokenize
import torch.nn.functional as F
import torch

from allennlp.common.file_utils import cached_path
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor
from allennlp.modules.token_embedders.embedding import EmbeddingsTextFile
from allennlp.data.instance import Instance
from allennlp.data.fields import TextField, SequenceLabelField, ListField
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer

from discourse.predictors.discourse_crf_predictor import DiscourseCRFClassifierPredictor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


MEDLINE_WORD_PATH = 'https://s3-us-west-2.amazonaws.com/pubmed-rct/medline_word_prob.json'
DISCOURSE_MODEL_PATH = 'https://s3-us-west-2.amazonaws.com/pubmed-rct/model_crf.tar.gz'
PUBMED_PRETRAINED_PATH = 'https://s3-us-west-2.amazonaws.com/pubmed-rct/wikipedia-pubmed-and-PMC-w2v.txt.gz'

p_dict = json.load(open(cached_path(MEDLINE_WORD_PATH), 'r'))
archive = load_archive(DISCOURSE_MODEL_PATH) # discourse model
discourse_predictor = Predictor.from_archive(archive, 'discourse_crf_classifier')


def read_json(file_path):
    """
    Read list from JSON path
    """
    if not os.path.exists(file_path):
        return []
    else:
        with open(file_path, 'r') as fp:
            ls = [json.loads(line) for line in fp]
        return ls


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


def get_sentence_vector(sent, embeddings, a=10e-4):
    """Average word vector for given list of words using fastText
    ref: A Simple but Tough-to-Beat Baseline for Sentence Embeddings, ICLR 2017
    https://openreview.net/pdf?id=SyK00v5xx
    """
    words = word_tokenize(sent)
    wvs = []
    for w in words:
        wv = embeddings.get(w, None)
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


if __name__ == '__main__':
    # read training dataset in JSON format
    # annotation: {}
    annotation_dataset = read_json('annotation_dataset.json')
    embeddings = read_embedding()
    label_map = {'CLAIM': 1, 'NOT CLAIM': 0}

    # create features
    features, labels = [], []
    for annotation in annotation_dataset:
        logits = torch.FloatTensor(discourse_predictor.predict_instance(text_to_instance(annotation['sentences']))['logits'])
        discourse_prob = F.softmax(logits, dim=-1)
        sents_embedding = torch.stack([torch.FloatTensor(get_sentence_vector(sent, embeddings, a=10e-4)) 
                                       for sent in annotation['sentences']])
        feature = torch.cat((sents_embedding, discourse_prob), dim=1)
        label = torch.IntTensor([label_map[l] for l in annotation['labels']])
        features.append(feature)
        labels.append(label)
    X = torch.cat(features)
    y = torch.cat(labels)
    
    logist_model = LogisticRegression()
    logist_model.fit(X, y)
    y_pred = logist_model.predict(X)
    print(precision_recall_fscore_support(y, y_pred, average='macro'))
    print(accuracy_score(y, y_pred))