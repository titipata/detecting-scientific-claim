import os
import sys
import csv
import numpy as np
from nltk import word_tokenize, sent_tokenize
from utils import *

import flask
import flask_login
from flask import Flask, request, session


app = Flask(__name__,
            template_folder='flask_templates')
app.secret_key = 'made in Thailand.'
app.config['TEMPLATES_AUTO_RELOAD'] = True
login_manager = flask_login.LoginManager()
login_manager.init_app(app)
with open(os.path.join('data', 'pmids.txt'), 'r') as f:
    pmids = [l.strip() for l in f.readlines()]


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
    pmid = pmids[0]
    return flask.redirect('/paper_id/%s' % pmid)


@app.route('/paper_id/<paper_id>')
def tag_paper_id(paper_id):
    paper_id = str(paper_id)
    data = parse_pubmed_xml(paper_id)
    data.update({'paper_id': paper_id,
                 'enumerate': enumerate,
                 'zip': zip,
                 'sentences': sent_tokenize(data.get('abstract'))})
    return flask.render_template('article.html', **data)


@app.route('/handle_submit/', methods=['GET', 'POST'])
def handle_submit():
    return flask.redirect('/paper_id/%s' % np.random.choice(pmids))


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
