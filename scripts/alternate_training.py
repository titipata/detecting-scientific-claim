"""
Script for alternate training
"""
import sys
sys.path.insert(0, '..')
from typing import Iterator, List, Dict, Optional
import json
import numpy as np
from itertools import chain
from overrides import overrides
from sklearn.model_selection import train_test_split

import torch
import torch.optim as optim
from torch.nn import ModuleList
import torch.nn.functional as F

from discourse.predictors import DiscourseClassifierPredictor
from discourse.dataset_readers import PubmedRCTReader, ClaimAnnotationReaderCSV
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
from allennlp.data.iterators import BucketIterator
from allennlp.training.trainer import Trainer
from allennlp.training.learning_rate_schedulers import LearningRateScheduler

from allennlp.modules.token_embedders import Embedding
from allennlp.modules.token_embedders.embedding import _read_embeddings_from_text_file
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import FeedForward, Seq2VecEncoder, TextFieldEmbedder
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder, PytorchSeq2VecWrapper
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy


EMBEDDING_DIM = 200
HIDDEN_DIM = 200
TRAIN_PATH = 'https://s3-us-west-2.amazonaws.com/pubmed-rct/train_labels.csv'
VALIDATION_PATH = 'https://s3-us-west-2.amazonaws.com/pubmed-rct/validation_labels.csv'
DISCOURSE_TRAIN_PATH = 'https://s3-us-west-2.amazonaws.com/pubmed-rct/train.json'
DISCOURSE_VALIDATION_PATH = 'https://s3-us-west-2.amazonaws.com/pubmed-rct/dev.json'
PUBMED_PRETRAINED_FILE = "https://s3-us-west-2.amazonaws.com/pubmed-rct/wikipedia-pubmed-and-PMC-w2v.txt.gz"


class DiscourseClaimClassifier(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 sentence_encoder: Seq2VecEncoder,
                 feedforward_discourse: FeedForward,
                 feedforward_claim: FeedForward,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(DiscourseClaimClassifier, self).__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.num_classes = self.vocab.get_vocab_size("labels")
        self.sentence_encoder = sentence_encoder
        self.feedforward_discourse = feedforward_discourse
        self.feedforward_claim = feedforward_claim
        self.metrics = {
            "accuracy": CategoricalAccuracy(),
            "accuracy3": CategoricalAccuracy(top_k=3)
        }
        self.loss = torch.nn.CrossEntropyLoss()
        initializer(self)

    @overrides
    def forward(self,
                sentence: Dict[str, torch.LongTensor],
                label: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        embedded_sentence = self.text_field_embedder(sentence)
        sentence_mask = util.get_text_field_mask(sentence)
        encoded_sentence = self.sentence_encoder(embedded_sentence, sentence_mask)
        
        if self.num_classes == 5:
            logits = self.feedforward_discourse(encoded_sentence)
        elif self.num_classes == 2:
            logits = self.feedforward_claim(encoded_sentence)
        else:
            ValueError('Number of classes should be either 2 for claims for 5 for discourse')

        output_dict = {'logits': logits}
        if label is not None:
            loss = self.loss(logits, label.squeeze(-1))
            for metric in self.metrics.values():
                metric(logits, label.squeeze(-1))
            output_dict["loss"] = loss

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        class_probabilities = F.softmax(output_dict['logits'])
        output_dict['class_probabilities'] = class_probabilities

        predictions = class_probabilities.cpu().data.numpy()
        argmax_indices = np.argmax(predictions, axis=-1)
        labels = [self.vocab.get_token_from_index(x, namespace='labels')
                  for x in argmax_indices]
        output_dict['label'] = labels
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}


if __name__ == '__main__':
    claim_reader = ClaimAnnotationReaderCSV()
    discourse_reader = PubmedRCTReader()
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
    feedforward_discourse = torch.nn.Sequential(torch.nn.Linear(2 * HIDDEN_DIM, 200), 
                                                torch.nn.Dropout(p=0.3), 
                                                torch.nn.Linear(200, 5))
    feedforward_claim = torch.nn.Sequential(torch.nn.Dropout(p=0.2),
                                            torch.nn.Linear(2 * HIDDEN_DIM, 2))
    model = DiscourseClaimClassifier(
        vocab,
        word_embeddings,
        sentence_encoder,
        feedforward_discourse,
        feedforward_claim
    )
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    lr_params = Params({"type": "reduce_on_plateau", "mode": "max", "factor": 0.5, "patience": 5})
    lr_scheduler = LearningRateScheduler.from_params(optimizer, lr_params)
    iterator = BucketIterator(batch_size=64, 
                              sorting_keys=[("sentence", "num_tokens")])
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
    torch.save(model.state_dict(), './model_alternate_training.th')