import os
import sys
import csv
import json
import yaml
import numpy as np
from nltk import sent_tokenize
from utils import parse_pubmed_xml, read_json, save_json

import flask
import flask_login
from flask import Flask, request, session


params = yaml.load(open("params.yaml", "r"))
PMIDS_PATH = params['pmids_path']
OUTPUT_PATH = params['output_path']
STORE_DETAILS = params['store_details']

if PMIDS_PATH.lower().endswith('.txt'):
    with open(PMIDS_PATH, 'r') as f:
        pmids = [l.strip() for l in f.readlines()]
elif PMIDS_PATH.lower().endswith('.json'):
    pmids_json = read_json(PMIDS_PATH)
    pmids = [str(p['paper_id']) for p in pmids_json]
    pmids_json_map = {int(p['paper_id']): p for p in pmids_json} # map from pmid to details
NUM_PMIDS = len(pmids)

app = Flask(__name__,
            template_folder='flask_templates')
app.secret_key = 'made in Thailand.'
app.config['TEMPLATES_AUTO_RELOAD'] = True
login_manager = flask_login.LoginManager()
login_manager.init_app(app)


def check_ids(collected_data, user_id, tagged=False):
    """
    Check PMIDs that are tagged or not tagged
    """
    pmids_tagged = [c['paper_id'] for c in collected_data
                    if c['user_id'] == user_id]
    if tagged is True:
        return pmids_tagged
    else:
        pmids_untagged = list(set(pmids) - set(pmids_tagged))
        return pmids_untagged


def remove_previous(collected_data, user_id, paper_id):
    return [c for c in collected_data
            if c['user_id'] is not user_id and c['paper_id'] is not paper_id]


class User(flask_login.UserMixin):
    pass


@login_manager.user_loader
def user_loader(email):
    if email not in users:
        return
    user = User()
    user.id = email
    return user


@login_manager.request_loader
def request_loader(request):
    email = request.form.get('email')
    user = User()
    user.id = email
    return user


@app.route('/login/', methods=['GET'])
def login():
    if 'email' in request.args:
        session['email'] = request.args['email']
        return flask.redirect('/')
    else:
        return flask.render_template('login.html')


@app.route('/logout/', methods=['GET'])
def logout():
    del session['email']
    return flask.redirect('/')


@app.route("/", methods=['GET', 'POST'])
def index():
    with open('fixtures.txt', 'r') as f:
        examples = [line for line in csv.reader(f, delimiter=',')]
    data = {'examples': examples}
    # progress
    if session.get('email') is not None:
        collected_data = read_json(OUTPUT_PATH)
        pmids_tagged = check_ids(collected_data, session['email'], tagged=True)
        data.update({'n_tagged': len(pmids_tagged),
                     'n_total': NUM_PMIDS})
    return flask.render_template('index.html', **data)


@app.route("/start_tagging/", methods=['POST'])
def start_tagging():
    collected_data = read_json(OUTPUT_PATH)
    pmids_untagged = check_ids(collected_data, session['email'], tagged=False)
    return flask.redirect('/paper_id/%s' % np.random.choice(pmids_untagged))


@app.route('/paper_id/<paper_id>')
def tag_paper_id(paper_id):
    """
    Tag the given PMID (paper_id)
    """
    if PMIDS_PATH.lower().endswith('.txt'):
        data = parse_pubmed_xml(paper_id)
        sentences = sent_tokenize(data['abstract'])
        data.update({
            'paper_id': paper_id,
            'sentences': sentences,
            'enumerate': enumerate,
            'zip': zip,
        })
    elif PMIDS_PATH.lower().endswith('.json'):
        data = pmids_json_map[int(paper_id)]
        data['paper_id'] = str(data['paper_id'])
        data.update({
            'enumerate': enumerate,
            'zip': zip,
        })
    # progress
    collected_data = read_json(OUTPUT_PATH)
    pmids_tagged = check_ids(collected_data, session['email'], tagged=True)
    data.update({'n_tagged': len(pmids_tagged),
                 'n_total': NUM_PMIDS})
    return flask.render_template('article.html', **data)


@app.route('/handle_submit/', methods=['GET', 'POST'])
def handle_submit():
    """
    Save tagged labels to JSON file
    """
    labels = [int(label) for label in request.form.getlist('labels')]
    user_id = session.get('email', '')
    if PMIDS_PATH.lower().endswith('.txt'):
        paper_id = request.form['paper_id']
        data = parse_pubmed_xml(paper_id)
        sentences = sent_tokenize(data['abstract'])
        data.update({
            'paper_id': paper_id,
            'user_id': user_id,
            'sentences': sentences,
            'labels': [int(s in labels) for s in np.arange(len(sentences))]
        })
    elif PMIDS_PATH.lower().endswith('.json'):
        paper_id = request.form['paper_id']
        data = pmids_json_map[int(paper_id)]
        data['paper_id'] = str(data['paper_id'])
        data.update({
            'user_id': user_id,
            'labels': [int(s in labels) for s in np.arange(len(data['sentences']))]
        })
        data.pop('enumerate', None)
        data.pop('zip', None)
        if STORE_DETAILS != 1:
            data.pop('title', None)
            data.pop('abstract', None)
            data.pop('sentences', None)
    # save data
    collected_data = read_json(OUTPUT_PATH)
    collected_data = remove_previous(collected_data,
                                     data['user_id'],
                                     data['paper_id'])
    collected_data += [data]
    save_json(collected_data, OUTPUT_PATH)

    pmids_untagged = check_ids(collected_data, session['email'], tagged=False)
    if len(pmids_untagged) > 0:
        return flask.redirect('/paper_id/%s' % np.random.choice(pmids_untagged))
    else:
        return flask.redirect('/')


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', thread=True)
