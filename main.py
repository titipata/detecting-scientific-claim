"""
Flask application for serving demo for detecting scientific claims
"""
import os
import sys
import json
from itertools import chain
import numpy as np
import pandas as pd
from nltk import word_tokenize, sent_tokenize

from lxml import etree, html
from urllib.request import urlopen

import flask
from flask import Flask, request
from gevent.wsgi import WSGIServer

PUBMED_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&retmode=xml&id=%s"
TESTING = False # if true, run testing
if not TESTING:
    from fastText import load_model
    from sklearn.externals import joblib

    from discourse.predictors import DiscourseClassifierPredictor
    from discourse.dataset_readers import PubmedRCTReader
    from discourse.models import DiscourseClassifier

    from allennlp.models.archival import load_archive
    from allennlp.service.predictors import Predictor
    from allennlp.common.file_utils import cached_path

    archive = load_archive('https://s3-us-west-2.amazonaws.com/pubmed-rct/model.tar.gz') # discourse model
    predictor = Predictor.from_archive(archive, 'discourse_classifier')

    MEDLINE_WORD_PATH = 'https://s3-us-west-2.amazonaws.com/pubmed-rct/medline_word_prob.json'
    LOGIST_PATH = 'https://s3-us-west-2.amazonaws.com/pubmed-rct/logist_model.pkl'
    PRINCIPAL_COMP_PATH = 'https://s3-us-west-2.amazonaws.com/pubmed-rct/principal_comp.txt'

    assert os.path.exists('wiki.en.bin') == True
    ft_model = load_model('wiki.en.bin') # fastText word vector
    p_dict = json.load(open(cached_path(MEDLINE_WORD_PATH), 'r')) # medline word probability
    logist_model = joblib.load(cached_path(LOGIST_PATH)) # trained logistic regression
    pc = np.loadtxt(cached_path(PRINCIPAL_COMP_PATH)) # principal comp


def get_sentence_vector(sent, a=10e-4):
    """Average word vector for given list of words using fastText

    ref: A Simple but Tough-to-Beat Baseline for Sentence Embeddings, ICLR 2017
    https://openreview.net/pdf?id=SyK00v5xx
    """
    words = word_tokenize(sent)
    wvs = []
    for w in words:
        wv = ft_model.get_word_vector(w)
        if w is not None:
            wvs.append([wv, p_dict.get(w, 0)])
    wvs = np.array(wvs)
    return np.mean(wvs[:, 0] * (a / (a + wvs[:, 1])), axis=0)


def remove_pc(x, pc):
    """
    Remove first principal component from an array
    """
    return x - x.dot(pc.transpose()) * pc


def parse_pubmed_xml(pmid):
    """
    Parse article information for the given PMID
    """
    url = PUBMED_URL % pmid
    page = urlopen(url).read()
    tree = html.fromstring(page)
    abstract = ''
    for e in tree.xpath('//abstract/abstracttext'):
        if e is not None:
            abstract += e.text
    title = ' '.join([e.text for e in tree.xpath('//articletitle')
                     if e is not None])
    return {'title': title, 'abstract': abstract}


def check_text_input(text_input):
    if 'www.ncbi.nlm.nih.gov/pubmed/' in text_input or text_input.isdigit():
        pmid = ''.join(c for c in text_input if c.isdigit())
        article = parse_pubmed_xml(pmid)
    else:
        article = {'title': '', 'abstract': text_input}
    return article


def stringify_children(node):
    """
    Filters and removes possible Nones in texts and tails
    ref: http://stackoverflow.com/questions/4624062/get-all-text-inside-a-tag-in-lxml
    """
    parts = ([node.text] +
             list(chain(*([c.text, c.tail] for c in node.getchildren()))) +
             [node.tail])
    return ''.join(filter(None, parts))


app = Flask(__name__,
            template_folder='flask_templates')
app.secret_key = 'made in Thailand.'
app.config['TEMPLATES_AUTO_RELOAD'] = True

@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text_input = request.form.get("text_input")
        article = check_text_input(text_input)
        if len(article.get('abstract', '').strip()) > 0:
            sentences = []
            sentences_vect = []
            labels = []

            if not TESTING:
                for sent in sent_tokenize(article.get('abstract', '')):
                    sent_vec = get_sentence_vector(sent)
                    sent_vec = remove_pc(sent_vec, pc) # remove the first principal component
                    discourse_output = predictor.predict_json({'sentence': sent})
                    label = discourse_output['label']
                    discourse_prob = np.array(discourse_output['class_probabilities'])
                    sentence_vect = np.hstack((sent_vec, discourse_prob))
                    sentences.append(sent)
                    sentences_vect.append(sentence_vect)
                    labels.append(label)
                sentences_vect = np.atleast_2d(np.vstack(sentences_vect))
                p_claims = 100 * logist_model.predict_proba(sentences_vect)[:, 1]
                p_claims = list(p_claims)
            else:
                for sent in sent_tokenize(abstract):
                    sentences.append(sent)
                    label = np.random.choice(['RESULTS', 'METHODS',
                                              'CONCLUSIONS', 'BACKGROUND',
                                              'OBJECTIVE'])
                    labels.append(label)
                p_claims = 100 * np.random.rand(len(sentences))
                p_claims = list(p_claims)
        else:
            sentences = ["Check input abstract, maybe abstract doesn't exist."]
            p_claims = [0]
            labels = ['NO LABEL']
        data = {'sents': sentences,
                'scores': p_claims,
                'labels': labels,
                'len': len,
                'enumerate': enumerate,
                'zip': zip}
        data.update(article)
    else:
        data = {'sents': [],
                'scores': [],
                'labels': [],
                'len': len,
                'enumerate': enumerate,
                'zip': zip}
        data.update({'title': '', 'abstract': ''})
    return flask.render_template('index.html', **data)


if __name__ == "__main__":
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
