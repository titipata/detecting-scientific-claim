from typing import Dict, Optional, List, Any

import numpy as np
from overrides import overrides
import torch
from torch.nn.modules.linear import Linear
import torch.nn.functional as F

from allennlp.common import Params
from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.modules import Seq2VecEncoder, TimeDistributed, TextFieldEmbedder, ConditionalRandomField, FeedForward
from allennlp.modules.conditional_random_field import allowed_transitions
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util
from allennlp.training.metrics import SpanBasedF1Measure, CategoricalAccuracy


@Model.register("discourse_crf_classifier")
class DiscourseCrfClassifier(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 sentence_encoder: Seq2VecEncoder,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 dropout: Optional[float] = None,
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(DiscourseCrfClassifier, self).__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.num_classes = self.vocab.get_vocab_size("labels")
        self.sentence_encoder = sentence_encoder
        if dropout:
            self.dropout = torch.nn.Dropout(dropout)
        else:
            self.dropout = None
        self.metrics = {
            "accuracy": CategoricalAccuracy(),
            "accuracy3": CategoricalAccuracy(top_k=3)
        }
        self.loss = torch.nn.CrossEntropyLoss()
        self.label_projection_layer = TimeDistributed(Linear(self.sentence_encoder.get_output_dim(), 
                                                             self.num_classes))
        
        constraints = None # allowed_transitions(label_encoding, labels)
        self.crf = ConditionalRandomField(
            self.num_classes, constraints,
            include_start_end_transitions=False
        )
        initializer(self)

    @overrides
    def forward(self,
                sentences: Dict[str, torch.LongTensor],
                labels: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        
        # print(sentences['tokens'].size())
        # print(labels.size())

        embedded_sentences = self.text_field_embedder(sentences)
        token_masks = util.get_text_field_mask(sentences, 1)
        sentence_masks = util.get_text_field_mask(sentences)

        # get sentence embedding
        encoded_sentences = []
        n_sents = embedded_sentences.size()[1] # size: (n_batch, n_sents, n_tokens, n_embedding)
        for i in range(n_sents):
            encoded_sentences.append(self.sentence_encoder(embedded_sentences[:, i, :, :], token_masks[:, i, :]))
        encoded_sentences = torch.stack(encoded_sentences, 1)

        # dropout layer
        if self.dropout:
            encoded_sentences = self.dropout(encoded_sentences)

        # print(encoded_sentences.size()) # size: (n_batch, n_sents, n_embedding)

        # CRF prediction
        logits = self.label_projection_layer(encoded_sentences) # size: (n_batch, n_sents, n_classes)
        best_paths = self.crf.viterbi_tags(logits, sentence_masks)
        predicted_labels = [x for x, y in best_paths]

        output_dict = {
            "logits": logits, 
            "mask": sentence_masks, 
            "labels": predicted_labels
        }
        
        # referring to https://github.com/allenai/allennlp/blob/master/allennlp/models/crf_tagger.py#L229-L239
        if labels is not None:
            log_likelihood = self.crf(logits, labels, sentence_masks)
            output_dict["loss"] = -log_likelihood

            class_probabilities = logits * 0.
            for i, instance_labels in enumerate(predicted_labels):
                for j, label_id in enumerate(instance_labels):
                    class_probabilities[i, j, label_id] = 1

            for metric in self.metrics.values():
                metric(class_probabilities, labels, sentence_masks.float())

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Coverts tag ids to actual tags.
        """
        output_dict["labels"] = [
            [self.vocab.get_token_from_index(label, namespace='labels')
                 for label in instance_labels]
                for instance_labels in output_dict["labels"]
        ]
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}