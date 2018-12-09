"""
Transfer learning for claim prediction using Discourse CRF model
"""
import sys
sys.path.insert(0, '..')

from typing import Iterator, List, Dict, Optional
import os
import json
import numpy as np
import pandas as pd
from itertools import chain
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from allennlp.common.util import JsonDict

import torch
import torch.optim as optim
from torch.nn import ModuleList
import torch.nn.functional as F

from discourse import read_json
from discourse.predictors import DiscourseClassifierPredictor
from discourse.dataset_readers import ClaimAnnotationReaderJSON
from discourse.models import DiscourseClassifier

from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor
from allennlp.common import Params
from allennlp.common.file_utils import cached_path

from allennlp.data.fields import Field, TextField, LabelField, ListField, SequenceLabelField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.vocabulary import Vocabulary

from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.iterators import BucketIterator, BasicIterator
from allennlp.training.trainer import Trainer
from allennlp.training.learning_rate_schedulers import LearningRateScheduler

from allennlp.modules import Seq2VecEncoder, TimeDistributed, TextFieldEmbedder, ConditionalRandomField, FeedForward
from torch.nn.modules.linear import Linear


EMBEDDING_DIM = 200
TRAIN_PATH = 'https://s3-us-west-2.amazonaws.com/pubmed-rct/train_labels.json'
VALIDATION_PATH = 'https://s3-us-west-2.amazonaws.com/pubmed-rct/validation_labels.json'
TEST_PATH = 'https://s3-us-west-2.amazonaws.com/pubmed-rct/test_labels.json'
DISCOURSE_MODEL_PATH = 'https://s3-us-west-2.amazonaws.com/pubmed-rct/model_crf.tar.gz'
archive = load_archive(DISCOURSE_MODEL_PATH)
discourse_predictor = Predictor.from_archive(archive, 'discourse_crf_predictor')


class ClaimCrfPredictor(Predictor):
    """"
    Predictor wrapper for the AcademicPaperClassifier
    """
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        sentences = json_dict['sentences']
        instance = self._dataset_reader.text_to_instance(sents=sentences)
        return instance


if __name__ == '__main__':
    # load model and freeze all layers
    model = discourse_predictor._model
    for param in list(model.parameters()):
        param.requires_grad = False
    num_classes, constraints, include_start_end_transitions = 2, None, False
    model.classifier_feedforward._linear_layers = ModuleList([torch.nn.Linear(2 * EMBEDDING_DIM, EMBEDDING_DIM), 
                                                              torch.nn.Linear(EMBEDDING_DIM, num_classes)])
    model.crf = ConditionalRandomField(num_classes, constraints, 
                                       include_start_end_transitions=include_start_end_transitions)
    model.label_projection_layer = TimeDistributed(Linear(2 * EMBEDDING_DIM, num_classes))

    reader = ClaimAnnotationReaderJSON()
    train_dataset = reader.read(TRAIN_PATH)
    validation_dataset = reader.read(VALIDATION_PATH)
    test_dataset = reader.read(TEST_PATH)
    vocab = discourse_predictor._model.vocab
    vocab._token_to_index['labels'] = {'0': 0, '1': 1}

    optimizer = optim.SGD(model.parameters(), lr=0.001)
    iterator = BasicIterator(batch_size=64)
    iterator.index_with(vocab)

    # unfreeze top layers and train
    for param in list(model.parameters())[:-4]:
        param.requires_grad = False
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        iterator=iterator,
        validation_iterator=iterator,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        patience=3,
        num_epochs=100, 
        cuda_device=-1
    )
    trainer.train()

    # unfreeze most layers and continue training
    for param in list(model.parameters())[1:]:
        param.requires_grad = True
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        iterator=iterator,
        validation_iterator=iterator,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        patience=3,
        num_epochs=100,
        cuda_device=-1
    )
    trainer.train()

    # precision, recall, f-score on validation set
    validation_list = read_json(cached_path(VALIDATION_PATH))
    claim_predictor = ClaimCrfPredictor(model, dataset_reader=reader)
    y_pred, y_true = [], []
    for val in validation_list:
        pred = claim_predictor.predict_json(val)
        logits = torch.FloatTensor(pred['logits'])
        best_paths = model.crf.viterbi_tags(torch.FloatTensor(pred['logits']).unsqueeze(0), 
                                            torch.LongTensor(pred['mask']).unsqueeze(0))
        predicted_labels = best_paths[0][0]
        y_pred.extend(predicted_labels)
        y_true.extend(val['labels'])
    y_true = np.array(y_true).astype(int)
    y_pred = np.array(y_pred).astype(int)
    print(precision_recall_fscore_support(y_true, y_pred, average='binary'))