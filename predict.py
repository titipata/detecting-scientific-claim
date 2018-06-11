import os
import json

from discourse.predictors import DiscourseClassifierPredictor
from discourse.dataset_readers import PubmedRCTReader
from discourse.models import DiscourseClassifier

from allennlp.models.archival import load_archive
from allennlp.service.predictors import Predictor

archive = load_archive('model.tar.gz')
predictor = Predictor.from_archive(archive, 'discourse_classifier')

if __name__ == '__main__':
    """
    Example of discourse prediction using ``discourse`` library
    created using AllenNLP
    """
    fixture_path = os.path.join('pubmed-rct', 'PubMed_200k_RCT', 'fixtures.json')
    json_sentences = [json.loads(line) for line in open(fixture_path, 'r').readlines()]
    for json_sentence in json_sentences:
        output = predictor.predict_json(json_sentence)
        output['sentence'] = json_sentence['sentence']
        print(output)
