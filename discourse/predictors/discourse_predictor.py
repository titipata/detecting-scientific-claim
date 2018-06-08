from typing import Tuple

from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.service.predictors.predictor import Predictor

@Predictor.register('discourse')
class DiscourseClassifierPredictor(Predictor):
    """"
    Predictor wrapper for the DiscourseClassifier
    """
    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Tuple[Instance, JsonDict]:
        sentence = json_dict['sentence']
        pos = json_dict['pos']
        instance = self._dataset_reader.text_to_instance(sent=sentence, pos=pos)

        # label_dict will be like {0: "ACL", 1: "AI", ...}
        label_dict = self._model.vocab.get_index_to_token_vocabulary('labels')
        # Convert it to list ["ACL", "AI", ...]
        all_labels = [label_dict[i] for i in range(len(label_dict))]

        return instance, {"all_labels": all_labels}
