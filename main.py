"""
Flask application for 'detecting scientific claims' demo
"""
import os
import sys
import json
from itertools import chain
import torch
from torch.nn import ModuleList, Linear
import torch.nn.functional as F
import numpy as np
import pandas as pd
from nltk import word_tokenize, sent_tokenize

from lxml import etree, html
import urllib

import flask
from flask import Flask, request
from gevent.pywsgi import WSGIServer

from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor
from allennlp.common.file_utils import cached_path
from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.modules import Seq2VecEncoder, TimeDistributed, TextFieldEmbedder, ConditionalRandomField, FeedForward

from discourse import read_json
from discourse.dataset_readers import ClaimAnnotationReaderJSON, CrfPubmedRCTReader
from discourse.predictors import DiscourseClassifierPredictor


TESTING = False # if true, run testing
EMBEDDING_DIM = 300
PUBMED_URL = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&retmode=xml&id=%s'
DISCOURSE_MODEL_PATH = 'https://detecting-scientific-claim.s3-us-west-2.amazonaws.com/model.tar.gz'
WEIGHT_PATH = 'https://detecting-scientific-claim.s3-us-west-2.amazonaws.com/model_crf_tf.th'


class ClaimCrfPredictor(Predictor):
    """"
    Predictor wrapper for the AcademicPaperClassifier
    """
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        sentences = json_dict['sentences']
        instance = self._dataset_reader.text_to_instance(sents=sentences)
        return instance


if not TESTING:

    archive = load_archive(DISCOURSE_MODEL_PATH)
    predictor = Predictor.from_archive(archive, 'discourse_crf_predictor')

    archive_ = load_archive(DISCOURSE_MODEL_PATH)
    discourse_predictor = Predictor.from_archive(archive_, 'discourse_crf_predictor')

    model = predictor._model
    for param in list(model.parameters()):
        param.requires_grad = False
    num_classes, constraints, include_start_end_transitions = 2, None, False
    model.crf = ConditionalRandomField(num_classes, constraints, 
                                       include_start_end_transitions=include_start_end_transitions)
    model.label_projection_layer = TimeDistributed(Linear(2 * EMBEDDING_DIM, num_classes))
    model.load_state_dict(torch.load(cached_path(WEIGHT_PATH), map_location='cpu'))
    reader = CrfPubmedRCTReader()
    claim_predictor = ClaimCrfPredictor(model, dataset_reader=reader)


def parse_pubmed_xml(pmid):
    """
    Parse article information for the given PMID
    """
    url = PUBMED_URL % pmid
    page = urllib.request.urlopen(url).read()
    tree = html.fromstring(page)
    abstract = ''
    for e in tree.xpath('//abstract/abstracttext'):
        if e is not None:
            abstract += stringify_children(e).strip()
    title = ' '.join([e.text for e in tree.xpath('//articletitle')
                     if e is not None])
    return {'title': title, 'abstract': abstract}


def check_text_input(text_input):
    """
    Check text input, if contains Pubmed URL, parse from Pubmed, 
    if not use text input as an abstract
    """
    if 'www.ncbi.nlm.nih.gov/pubmed/' in text_input.lower() or 'pubmed.ncbi.nlm.nih.gov' in text_input.lower() or text_input.isdigit():
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
            abstract = article.get('abstract', '')
            sentences = sent_tokenize(abstract)
            labels = []
            if not TESTING:
                discourse_output = discourse_predictor.predict_json({'abstract': abstract})
                labels = discourse_output['labels']
                pred = claim_predictor.predict_json({'sentences': sentences})
                best_paths = model.crf.viterbi_tags(torch.FloatTensor(pred['logits']).unsqueeze(0), 
                                                    torch.LongTensor(pred['mask']).unsqueeze(0))
                p_claims = 100 * np.array(best_paths[0][0])
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
    http_server = WSGIServer(('0.0.0.0', 5001), app)
    http_server.serve_forever()
