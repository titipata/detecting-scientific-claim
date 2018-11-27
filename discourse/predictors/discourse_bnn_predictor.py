from typing import Tuple

from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor

@Predictor.register('discourse_bnn_predictor')
class DiscourseBNNClassifierPredictor(Predictor):
    """"
    Predictor wrapper for the DiscourseBNNClassifier
    """
    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        sentence = json_dict['sentence']
        instance = self._dataset_reader.text_to_instance(sent=sentence)
        return instance
