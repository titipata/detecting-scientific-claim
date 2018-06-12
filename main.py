import os
import json
import numpy as np
import pandas as pd
import flask
from flask import Flask, request, flash

from nltk import word_tokenize, sent_tokenize
from fastText import load_model
from sklearn.externals import joblib

from discourse.predictors import DiscourseClassifierPredictor
from discourse.dataset_readers import PubmedRCTReader
from discourse.models import DiscourseClassifier

from allennlp.models.archival import load_archive
from allennlp.service.predictors import Predictor
from allennlp.common.file_utils import cached_path


app = Flask(__name__, template_folder='flask_templates')
app.secret_key = 'made in Thailand.'
app.config['TEMPLATES_AUTO_RELOAD'] = True

# discourse model
archive = load_archive('model.tar.gz')
predictor = Predictor.from_archive(archive, 'discourse_classifier')


p_dict = json.load(open(os.path.join('medline', 'medline_word_prob.json'), 'r')) # medline word probability
ft_model = load_model(os.path.join('medline', 'wiki.en.bin')) # fastText word vector
logist_model = joblib.load(os.path.join('medline', 'logist_model.pkl')) # trained logistic regression
pc = np.loadtxt(os.path.join('medline', 'principal_comp.txt')) # principal comp


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


@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        abstract = request.form.get("text_input")
        if len(abstract.strip()) > 0:
            sentences = []
            sentences_vect = []
            labels = []
            for sent in sent_tokenize(abstract):
                sent_vec = get_sentence_vector(sent)
                sent_vec = remove_pc(sent_vec, pc) # remove the first principal component
                discourse_output = predictor.predict_json({'sentence': sent})
                label = discourse_output['label']
                discourse_prob = np.array(discourse_output['class_probabilities'])
                sentence_vect = np.hstack((sent_vec, discourse_prob))
                # sentence_vect = discourse_prob
                sentences.append(sent)
                sentences_vect.append(sentence_vect)
                labels.append(label)
            sentences_vect = np.atleast_2d(np.vstack(sentences_vect))
            p_claims = 100 * logist_model.predict_proba(sentences_vect)[:, 1]
            p_claims = list(p_claims)
        else:
            sentences = ["Check input abstract, maybe abstract doesn't exist."]
            p_claims = [0]
            labels = ['']
        data = {'sents': sentences,
                'scores': p_claims,
                'labels': labels,
                'abstract': abstract,
                'len': len,
                'enumerate': enumerate,
                'zip': zip}
    else:
        data = {'sents': [],
                'scores': [],
                'labels': [],
                'len': len,
                'enumerate': enumerate,
                'zip': zip}
    return flask.render_template('index.html', **data)


if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', processes=3)
