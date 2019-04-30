import sys
sys.path.insert(0, '..')
import os
import re
import json
import requests
from bs4 import BeautifulSoup
import twitter
from biorxiv_cli import Client
import spacy

import torch
from torch.nn import ModuleList, Linear

from discourse.dataset_readers import ClaimAnnotationReaderJSON, CrfPubmedRCTReader
from discourse.predictors import DiscourseClassifierPredictor

from allennlp.data import Instance
from allennlp.common.util import JsonDict
from allennlp.models.archival import load_archive, cached_path
from allennlp.predictors import Predictor
from allennlp.modules import TimeDistributed, TextFieldEmbedder, ConditionalRandomField, FeedForward


nlp = spacy.load('en_core_web_sm')
api = twitter.Api(consumer_key = '',
                  consumer_secret = '',
                  access_token_key = '',
                  access_token_secret = '')
client = Client()

BIORXIV_PATH = 'biorxiv_urls.txt'
EMBEDDING_DIM = 300
DISCOURSE_MODEL_PATH = 'https://s3-us-west-2.amazonaws.com/pubmed-rct/model_crf.tar.gz'
WEIGHT_PATH = 'https://s3-us-west-2.amazonaws.com/pubmed-rct/model_crf_tf.th'


class ClaimCrfPredictor(Predictor):
    """"
    Predictor wrapper for the AcademicPaperClassifier
    """
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        sentences = json_dict['sentences']
        instance = self._dataset_reader.text_to_instance(sents=sentences)
        return instance


archive = load_archive(DISCOURSE_MODEL_PATH)
predictor = Predictor.from_archive(archive, 'discourse_crf_predictor')
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


def get_recent_biorxiv_tweets(count=10):
    """
    Get recent tweets from bioRxiv
    """
    tweets = api.GetUserTimeline(screen_name="biorxivpreprint", count=count)
    return tweets


def clean_abstract(abstract):
    abstract = abstract.replace('Objective: ', '')
    abstract = abstract.replace('OBJECTIVE: ', '')
    abstract = abstract.replace('Background: ', '')
    abstract = abstract.replace('BACKGROUND: ', '')
    abstract = abstract.replace('Method: ', '')
    abstract = abstract.replace('METHOD: ', '')
    abstract = abstract.replace('Result: ', '')
    abstract = abstract.replace('RESULT: ', '')
    abstract = abstract.replace('Conclusion: ', '')
    abstract = abstract.replace('CONCLUSION: ', '')
    return abstract


def tweet_biorxiv_claim():
    # read file
    if os.path.isfile(BIORXIV_PATH):
        with open(BIORXIV_PATH, 'r') as f:
            biorxiv_urls = json.loads(f.read())
    else:
        biorxiv_urls = []

    # get publications from biorxiv
    publications = client.read(['all'])
    for publication in publications:
        # title = publication['title']
        abstract = publication['summary']
        abstract = clean_abstract(abstract)
        if publication['id'] not in biorxiv_urls:
            sentences = [sent.string.strip() 
                         for sent in nlp(abstract).sents]
            pred = claim_predictor.predict_json({'sentences': sentences})
            best_paths = model.crf.viterbi_tags(torch.FloatTensor(pred['logits']).unsqueeze(0), 
                                                torch.LongTensor(pred['mask']).unsqueeze(0))
            claim_sentences = [s for s, p in zip(sentences, best_paths[0][0]) if int(p) == 1]
            if len(claim_sentences) >= 1:
                claim_sentence = claim_sentences[-1]
                tweet_text = claim_sentence[0: 230] + ' -- claim extracted from ' + publication['id']
                print(tweet_text)
                # api.PostUpdate(tweet_text)
            biorxiv_urls.append(publication['id'])
    
    # save tweeted urls
    with open(BIORXIV_PATH, 'w') as f:
        f.write(json.dumps(biorxiv_urls, indent=2))


