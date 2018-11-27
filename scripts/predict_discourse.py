"""
Example of discourse prediction using ``discourse`` library
created using AllenNLP
"""
import os
import sys
import json
sys.path.insert(0, '..')

from discourse.predictors import DiscourseClassifierPredictor
from discourse.dataset_readers import PubmedRCTReader
from discourse.models import DiscourseClassifier
from discourse import read_json

from allennlp.models.archival import load_archive
from allennlp.common.file_utils import cached_path
from allennlp.service.predictors import Predictor

MODEL_PATH = 'https://s3-us-west-2.amazonaws.com/pubmed-rct/model.tar.gz'
archive = load_archive(cached_path(MODEL_PATH))
predictor = Predictor.from_archive(archive, 'discourse_predictor')

if __name__ == '__main__':
    fixture_path = os.path.join('..', 'pubmed-rct', 'PubMed_200k_RCT', 'fixtures.json')
    json_sentences = read_json(fixture_path)
    for json_sentence in json_sentences:
        output = predictor.predict_json(json_sentence)
        output['sentence'] = json_sentence['sentence']
        print(output)
