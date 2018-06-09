from typing import Tuple

from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.service.predictors.predictor import Predictor

@Predictor.register('discourse_classifier')
class DiscourseClassifierPredictor(Predictor):
    """"
    Predictor wrapper for the DiscourseClassifier
    """
    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Tuple[Instance, JsonDict]:
        sentence = json_dict['sentence_text']
        instance = self._dataset_reader.text_to_instance(sent=sentence)

        # label_dict will be like {0: "RESULTS", 1: "METHODS", ...}
        label_dict = self._model.vocab.get_index_to_token_vocabulary('labels')

        # Convert it to list ["RESULTS", "METHODS", "CONCLUSIONS", "BACKGROUND", "OBJECTIVE"]
        all_labels = [label_dict[i] for i in range(len(label_dict))]

        return instance, {"all_labels": all_labels}
