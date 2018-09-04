from typing import Dict, Optional, Union

import numpy
from overrides import overrides
import torch
from torch import nn
import torch.nn.functional as F

from allennlp.common import Params
from allennlp.common.checks import check_dimensions_match, ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import Elmo, FeedForward, Maxout, Seq2SeqEncoder, TextFieldEmbedder
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy


@Model.register("discourse_bnn_classifier")
class DiscourseBNNClassifier(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 embedding_dropout: float,
                 pre_encode_feedforward: FeedForward,
                 encoder: Seq2SeqEncoder,
                 integrator: Seq2SeqEncoder,
                 integrator_dropout: float,
                 output_layer: FeedForward,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(DiscourseBNNClassifier, self).__init__(vocab, regularizer)

        self._text_field_embedder = text_field_embedder
        self._embedding_dropout = nn.Dropout(embedding_dropout)
        self._num_classes = self.vocab.get_vocab_size("labels")

        self._pre_encode_feedforward = pre_encode_feedforward
        self._encoder = encoder
        self._integrator = integrator
        self._integrator_dropout = nn.Dropout(integrator_dropout)
        self._combined_integrator_output_dim = self._integrator.get_output_dim()

        self._self_attentive_pooling_projection = nn.Linear(
                self._combined_integrator_output_dim, 1)
        self._output_layer = output_layer

        self.metrics = {
            "accuracy": CategoricalAccuracy(),
            "accuracy3": CategoricalAccuracy(top_k=3)
        }
        self.loss = torch.nn.CrossEntropyLoss()
        initializer(self)

    @overrides
    def forward(self,  # type: ignore
                sentence: Dict[str, torch.LongTensor],
                label: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        sentence_mask = util.get_text_field_mask(sentence).float()
        embedded_sentence = self._text_field_embedder(sentence)

        dropped_embedded_sent = self._embedding_dropout(embedded_sentence)
        pre_encoded_sent = self._pre_encode_feedforward(dropped_embedded_sent)
        encoded_tokens = self._encoder(pre_encoded_sent, sentence_mask)

        # Compute biattention. This is a special case since the inputs are the same.
        attention_logits = encoded_tokens.bmm(encoded_tokens.permute(0, 2, 1).contiguous())
        attention_weights = util.last_dim_softmax(attention_logits, sentence_mask)
        encoded_sentence = util.weighted_sum(encoded_tokens, attention_weights)

        # Build the input to the integrator
        integrator_input = torch.cat([encoded_tokens,
                                      encoded_tokens - encoded_sentence,
                                      encoded_tokens * encoded_sentence], 2)
        integrated_encodings = self._integrator(integrator_input, sentence_mask)

        # Simple Pooling layers
        max_masked_integrated_encodings = util.replace_masked_values(
                integrated_encodings, sentence_mask.unsqueeze(2), -1e7)
        max_pool = torch.max(max_masked_integrated_encodings, 1)[0]
        min_masked_integrated_encodings = util.replace_masked_values(
                integrated_encodings, sentence_mask.unsqueeze(2), +1e7)
        min_pool = torch.min(min_masked_integrated_encodings, 1)[0]
        mean_pool = torch.sum(integrated_encodings, 1) / torch.sum(sentence_mask, 1, keepdim=True)

        # Self-attentive pooling layer
        # Run through linear projection. Shape: (batch_size, sequence length, 1)
        # Then remove the last dimension to get the proper attention shape (batch_size, sequence length).
        self_attentive_logits = self._self_attentive_pooling_projection(integrated_encodings).squeeze(2)
        self_weights = util.masked_softmax(self_attentive_logits, sentence_mask)
        self_attentive_pool = util.weighted_sum(integrated_encodings, self_weights)

        pooled_representations = torch.cat([max_pool, min_pool, mean_pool, self_attentive_pool], 1)
        pooled_representations_dropped = self._integrator_dropout(pooled_representations).squeeze(1)

        logits = self._output_layer(pooled_representations_dropped)
        output_dict = {'logits': logits}
        if label is not None:
            loss = self.loss(logits, label.squeeze(-1))
            for metric in self.metrics.values():
                metric(logits, label.squeeze(-1))
            output_dict["loss"] = loss

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Does a simple argmax over the class probabilities, converts indices to string labels, and
        adds a ``"label"`` key to the dictionary with the result.
        """
        class_probabilities = F.softmax(output_dict['logits'], dim=-1)
        output_dict['class_probabilities'] = class_probabilities

        predictions = output_dict['class_probabilities'].cpu().data.numpy()
        argmax_indices = numpy.argmax(predictions, axis=-1)
        labels = [self.vocab.get_token_from_index(x, namespace="labels")
                  for x in argmax_indices]
        output_dict['label'] = labels
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}
