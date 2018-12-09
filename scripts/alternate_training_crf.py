"""
Script for alternate training with CRF layer
"""
import sys
sys.path.insert(0, '..')
from typing import Iterator, List, Dict, Optional
import json
import pandas as pd
from itertools import chain
from overrides import overrides
from sklearn.model_selection import train_test_split

import torch
import torch.optim as optim
from torch.nn import ModuleList, Linear
import torch.nn.functional as F

from discourse.predictors import DiscourseClassifierPredictor
from discourse.dataset_readers import CrfPubmedRCTReader, ClaimAnnotationReaderJSON
from discourse.models import DiscourseClassifier

from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor
from allennlp.common.file_utils import cached_path

from allennlp.data.fields import TextField, LabelField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.vocabulary import Vocabulary

from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.iterators import BasicIterator
from allennlp.training.trainer import Trainer
from allennlp.training.learning_rate_schedulers import LearningRateScheduler

from allennlp.modules.token_embedders import Embedding
from allennlp.modules.token_embedders.embedding import _read_embeddings_from_text_file
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import Seq2VecEncoder, TimeDistributed, TextFieldEmbedder, ConditionalRandomField, FeedForward
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder, PytorchSeq2VecWrapper
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy


EMBEDDING_DIM = 200
HIDDEN_DIM = 200
TRAIN_PATH = 'https://s3-us-west-2.amazonaws.com/pubmed-rct/train_labels.json'
VALIDATION_PATH = 'https://s3-us-west-2.amazonaws.com/pubmed-rct/validation_labels.json'
TEST_PATH = 'https://s3-us-west-2.amazonaws.com/pubmed-rct/test_labels.json'
DISCOURSE_TRAIN_PATH = 'https://s3-us-west-2.amazonaws.com/pubmed-rct/train.txt'
DISCOURSE_VALIDATION_PATH = 'https://s3-us-west-2.amazonaws.com/pubmed-rct/dev.txt'
PUBMED_PRETRAINED_FILE = "https://s3-us-west-2.amazonaws.com/pubmed-rct/wikipedia-pubmed-and-PMC-w2v.txt.gz"


class DiscourseClaimCrfClassifier(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 sentence_encoder: Seq2VecEncoder,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None, 
                 label_smoothing: float = None) -> None:
        super(DiscourseClaimCrfClassifier, self).__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.num_classes = self.vocab.get_vocab_size("labels")
        self.sentence_encoder = sentence_encoder
        self.metrics = {
            "accuracy": CategoricalAccuracy(),
            "accuracy3": CategoricalAccuracy(top_k=3)
        }
        self.loss = torch.nn.CrossEntropyLoss()
        self.label_projection_layer_discourse = TimeDistributed(Linear(self.sentence_encoder.get_output_dim(), 5))
        self.label_projection_layer_claim = TimeDistributed(Linear(self.sentence_encoder.get_output_dim(), 2))
        
        constraints = None
        self.crf_discourse = ConditionalRandomField(5, constraints, include_start_end_transitions=False)
        self.crf_claim = ConditionalRandomField(2, constraints, include_start_end_transitions=False)
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

        # CRF prediction
        if self.num_classes == 5:
            logits = self.label_projection_layer_discourse(encoded_sentences) # size: (n_batch, n_sents, n_classes)
            best_paths = self.crf_discourse.viterbi_tags(logits, sentence_masks)
            predicted_labels = [x for x, y in best_paths]
        else:
            logits = self.label_projection_layer_claim(encoded_sentences) # size: (n_batch, n_sents, n_classes)
            best_paths = self.crf_claim.viterbi_tags(logits, sentence_masks)
            predicted_labels = [x for x, y in best_paths]

        output_dict = {
            "logits": logits, 
            "mask": sentence_masks, 
            "labels": predicted_labels
        }
        
        # referring to https://github.com/allenai/allennlp/blob/master/allennlp/models/crf_tagger.py#L229-L239
        if labels is not None:
            if self.num_classes == 5:
                log_likelihood = self.crf_discourse(logits, labels, sentence_masks)
            else:
                log_likelihood = self.crf_claim(logits, labels, sentence_masks)
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


if __name__ == '__main__':
    claim_reader = ClaimAnnotationReaderJSON()
    discourse_reader = CrfPubmedRCTReader()
    claim_train_dataset = claim_reader.read(cached_path(TRAIN_PATH))
    claim_validation_dataset = claim_reader.read(cached_path(VALIDATION_PATH))
    discourse_train_dataset = discourse_reader.read(cached_path(DISCOURSE_TRAIN_PATH))
    discourse_validation_dataset = discourse_reader.read(cached_path(DISCOURSE_VALIDATION_PATH))
    vocab = Vocabulary.from_instances(claim_train_dataset + \
                                      claim_validation_dataset + \
                                      discourse_train_dataset + \
                                      discourse_validation_dataset)
    discourse_dict = {'RESULTS': 0, 'METHODS': 1, 'CONCLUSIONS': 2, 'BACKGROUND': 3, 'OBJECTIVE': 4}
    claim_dict = {'0': 0, '1': 1}
    embedding_matrix = _read_embeddings_from_text_file(file_uri=PUBMED_PRETRAINED_FILE, 
                                                    embedding_dim=EMBEDDING_DIM, 
                                                    vocab=vocab)
    token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'), 
                                embedding_dim=EMBEDDING_DIM,
                                weight=embedding_matrix)
    word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})
    sentence_encoder = PytorchSeq2VecWrapper(torch.nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, 
                                                        batch_first=True, bidirectional=True))
    model = DiscourseClaimCrfClassifier(
        vocab,
        word_embeddings,
        sentence_encoder,
    )
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    lr_params = Params({"type": "reduce_on_plateau", "mode": "max", "factor": 0.5, "patience": 5})
    lr_scheduler = LearningRateScheduler.from_params(optimizer, lr_params)
    iterator = BasicIterator(batch_size=64)
    iterator.index_with(vocab)

    for (l, train_dataset, validation_dataset, n_classes, num_epochs, patience) in [(discourse_dict, discourse_train_dataset, discourse_validation_dataset, 5, 20, 2), 
                                                                                    (claim_dict, claim_train_dataset, claim_validation_dataset, 2, 20, 2), 
                                                                                    (discourse_dict, discourse_train_dataset, claim_validation_dataset, 5, 10, 2), 
                                                                                    (claim_dict, claim_train_dataset, claim_validation_dataset, 2, 20, 2)]:
        model.vocab._token_to_index['labels'] = l
        model.num_classes = n_classes
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            learning_rate_scheduler=lr_scheduler,
            iterator=iterator,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            patience=patience,
            num_epochs=num_epochs, 
            cuda_device=0
        )
        trainer.train()
    # save trained weight
    torch.save(model.state_dict(), './model_alternate_training_crf.th')