from typing import Tuple

from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor

@Predictor.register('discourse_crf_classifier')
class DiscourseCRFClassifierPredictor(Predictor):
    """"
    Predictor wrapper for the DiscourseClassifier
    """
    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        abstract = json_dict["abstract"]
        sentences = [sent.text.strip() for sent in abstract.sents]
        instance = self._dataset_reader.text_to_instance(sentences=sentences)
        return instance
