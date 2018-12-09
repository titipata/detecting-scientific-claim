"""
Example script to predict claim using trained CRF claim model 
based on pre-trained CRF model from structured abstract
"""
import os
import sys
sys.path.insert(0, '..')

import numpy as np
from nltk import sent_tokenize
import torch
from torch.nn import ModuleList
from torch.nn.modules.linear import Linear

from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor
from allennlp.common.file_utils import cached_path
from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.modules import Seq2VecEncoder, TimeDistributed, TextFieldEmbedder, ConditionalRandomField, FeedForward

from discourse import read_json
from discourse.dataset_readers import ClaimAnnotationReaderJSON, CrfPubmedRCTReader
from discourse.predictors import DiscourseClassifierPredictor

DISCOURSE_MODEL_PATH = 'https://s3-us-west-2.amazonaws.com/pubmed-rct/model_crf.tar.gz'
WEIGHT_PATH = 'https://s3-us-west-2.amazonaws.com/pubmed-rct/model_crf_tf.th'
archive = load_archive(DISCOURSE_MODEL_PATH)
discourse_predictor = Predictor.from_archive(archive, 'discourse_crf_predictor')
EMBEDDING_DIM = 200


class ClaimCrfPredictor(Predictor):
    """"
    Predictor wrapper for the AcademicPaperClassifier
    """
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        sentences = json_dict['sentences']
        instance = self._dataset_reader.text_to_instance(sents=sentences)
        return instance


if __name__ == '__main__':
    model = discourse_predictor._model
    for param in list(model.parameters()):
        param.requires_grad = False
    num_classes, constraints, include_start_end_transitions = 2, None, False
    model.classifier_feedforward._linear_layers = ModuleList([torch.nn.Linear(2 * EMBEDDING_DIM, EMBEDDING_DIM), 
                                                              torch.nn.Linear(EMBEDDING_DIM, num_classes)])
    model.crf = ConditionalRandomField(num_classes, constraints, 
                                       include_start_end_transitions=include_start_end_transitions)
    model.label_projection_layer = TimeDistributed(Linear(2 * EMBEDDING_DIM, num_classes))
    model.load_state_dict(torch.load(cached_path(WEIGHT_PATH)))

    reader = CrfPubmedRCTReader()
    claim_predictor = ClaimCrfPredictor(model, dataset_reader=reader)

    fixture_path = os.path.join('..', 'pubmed-rct', 'PubMed_200k_RCT', 'fixtures_crf.json')
    examples = read_json(fixture_path)
    pred_list = []
    for example in examples:
        sentences = sent_tokenize(example['abstract'])
        instance = reader.text_to_instance(sents=sentences)
        pred = claim_predictor.predict_instance(instance)
        logits = torch.FloatTensor(pred['logits'])
        best_paths = model.crf.viterbi_tags(torch.FloatTensor(pred['logits']).unsqueeze(0), 
                                            torch.LongTensor(pred['mask']).unsqueeze(0))
        pred_list.append(best_paths[0][0])
    print(pred_list)
