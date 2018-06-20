import os
import sys
import csv
import json
import yaml
import numpy as np
from nltk import sent_tokenize
from utils import *

import flask
import flask_login
from flask import Flask, request, session


params = yaml.load(open("params.yaml", "r"))
PMIDS_PATH = params['pmids_path']
OUTPUT_PATH = params['output_path']
with open(PMIDS_PATH, 'r') as f:
    pmids = [l.strip() for l in f.readlines()]


app = Flask(__name__,
            template_folder='flask_templates')
app.secret_key = 'made in Thailand.'
app.config['TEMPLATES_AUTO_RELOAD'] = True
login_manager = flask_login.LoginManager()
login_manager.init_app(app)


def read_json(file_path):
    """
    Read collected file from path
    """
    if not os.path.exists(file_path):
        collected_data = []
        return collected_data
    else:
        with open(file_path, 'r') as fp:
            collected_data = [json.loads(line) for line in fp]
        return collected_data


def save_json(ls, file_path):
    """
    Save list of dictionary to JSON
    """
    with open(file_path, 'w') as fp:
        fp.write('\n'.join(json.dumps(i) for i in ls))


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
    data = parse_pubmed_xml(paper_id)
    sentences = sent_tokenize(data['abstract'])
    data.update({
        'paper_id': paper_id,
        'sentences': sentences,
        'enumerate': enumerate,
        'zip': zip,
    })
    return flask.render_template('article.html', **data)


@app.route('/handle_submit/', methods=['GET', 'POST'])
def handle_submit():
    """
    Save tagged labels to JSON file
    """
    labels = [int(label) for label in request.form.getlist('labels')]
    user_id = session.get('email', '')
    paper_id = request.form['paper_id']
    data = parse_pubmed_xml(paper_id)
    sentences = sent_tokenize(data['abstract'])
    data.update({
        'paper_id': paper_id,
        'user_id': user_id,
        'sentences': sentences,
        'labels': [int(s in labels) for s in np.arange(len(sentences))]
    })
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
    app.run(debug=True, host='0.0.0.0')
