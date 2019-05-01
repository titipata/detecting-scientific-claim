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
                  access_token_secret = '') # add API key here
client = Client() # for fetching biorxiv

BIORXIV_PATH = 'biorxiv_urls.txt'
BIORXIV_STATUS_PATH = 'biorxiv_status_ids.txt'
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


def parse_bioarxiv(url):
    """
    Parsing title and abstract from bioarxiv  
    return output dictionary
    """
    site = requests.get(url)
    soup = BeautifulSoup(site.content, 'html.parser')
    title = soup.find('h1', attrs={'class': 'highwire-cite-title'})
    title = title.text if title is not None else ''
    abstract = soup.find('div', attrs={'class': 'section abstract'})
    if abstract is not None:
        abstract = abstract.find('p')
    if abstract is not None:
        abstract = abstract.text if abstract is not None else ''
    else:
        abstract = ''
    return {
        'title': title, 
        'abstract': abstract
    }


def clean_abstract(abstract):
    """
    Clean a given abstract
    """
    abstract = abstract.replace('Objective: ', '')
    abstract = abstract.replace('OBJECTIVE: ', '')
    abstract = abstract.replace('Background: ', '')
    abstract = abstract.replace('BACKGROUND: ', '')
    abstract = abstract.replace('Method: ', '')
    abstract = abstract.replace('METHOD: ', '')
    abstract = abstract.replace('Result: ', '')
    abstract = abstract.replace('RESULT: ', '')
    abstract = abstract.replace('Conclusion ', '')
    abstract = abstract.replace('Conclusion: ', '')
    abstract = abstract.replace('CONCLUSION: ', '')
    return abstract


def detect_claim(abstract):
    """
    Detect given claim sentences from a given abstract
    """
    abstract = clean_abstract(abstract)
    sentences = [sent.string.strip() 
                for sent in nlp(abstract).sents]
    pred = claim_predictor.predict_json({'sentences': sentences})
    best_paths = model.crf.viterbi_tags(torch.FloatTensor(pred['logits']).unsqueeze(0), 
                                        torch.LongTensor(pred['mask']).unsqueeze(0))
    claim_sentences = [s for s, p in zip(sentences, best_paths[0][0]) if int(p) == 1]
    return claim_sentences


def tweet_biorxiv_claim(mode='tweet', verbose=True, testing=False):
    """
    Run this function to tweet or reply to the original tweet of Biorxiv
    
    mode: str, 'tweet' for tweeting claim from new articles, 
        'reply' to reply with extracted claim tweeted by @biorxivpreprint
    verbose: bool, if True, print the tweet text
    testing: bool, if True, will not tweet out
    """
    if mode == 'tweet':
        try:
            # read path file
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
                if publication['id'] not in biorxiv_urls:
                    claim_sentences = detect_claim(abstract)
                    if len(claim_sentences) >= 1:
                        claim_sentence = claim_sentences[-1]
                        tweet_text = '[extracted claim] ' + claim_sentence[0: 230] + ' ' + publication['id']
                        if not testing:
                            api.PostUpdate(tweet_text)
                        if verbose:
                            print(tweet_text)
                    biorxiv_urls.append(publication['id'])
            
            # save tweeted urls
            if not testing:
                with open(BIORXIV_PATH, 'w') as f:
                    f.write(json.dumps(biorxiv_urls, indent=2))
        except:
            pass

    elif mode == 'reply':
        try:
            # read status file
            if os.path.isfile(BIORXIV_STATUS_PATH):
                with open(BIORXIV_STATUS_PATH, 'r') as f:
                    biorxiv_status_ids = json.loads(f.read())
            else:
                biorxiv_status_ids = []
        
            tweets = get_recent_biorxiv_tweets(count=5)
            for tweet in tweets:
                expanded_url = tweet.urls[0].expanded_url
                if expanded_url not in biorxiv_status_ids and 'biorxiv' in expanded_url:
                    d = parse_bioarxiv(expanded_url)
                    abstract = d.get('abstract', '')
                    claim_sentences = detect_claim(abstract)
                    if len(claim_sentences) >= 1:
                        claim_sentence = claim_sentences[-1]
                        tweet_text = '[extracted claim] ' + claim_sentence[0: 240] + ' @biorxivpreprint'
                        if not testing:
                            api.PostUpdate(tweet_text, in_reply_to_status_id=tweet.id)
                        if verbose:
                            print(tweet_text)
                biorxiv_status_ids.append(tweet.id)
            
            # save tweeted status
            if not testing:
                with open(BIORXIV_STATUS_PATH, 'w') as f:
                    f.write(json.dumps(biorxiv_status_ids, indent=2))
        except:
            pass
    
    else:
        print('Choose mode ``tweet`` or ``reply``!')