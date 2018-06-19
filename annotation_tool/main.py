import os
import sys
from nltk import word_tokenize, sent_tokenize

import csv
from lxml import etree, html
from urllib.request import urlopen

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
def tag_paper_id(paper_id, query_from_s2=False):
    return flask.render_template('article.html', paper_id=paper_id)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
