import spacy
from typing import Tuple
from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor

nlp = spacy.load('en_core_web_sm')

@Predictor.register('discourse_crf_predictor')
class DiscourseCRFClassifierPredictor(Predictor):
    """"
    Predictor wrapper for the DiscourseClassifier
    """
    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        abstract = json_dict["abstract"]
        abstract = nlp(abstract)
        sentences = [sent.text.strip() for sent in abstract.sents]
        instance = self._dataset_reader.text_to_instance(sents=sentences)
        return instance
