"""
Transfer learning for claim prediction using Discourse model
"""
import sys
sys.path.insert(0, '..')
from typing import Iterator, List, Dict, Optional
import numpy as np
import pandas as pd
from itertools import chain
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_recall_fscore_support

import torch
import torch.optim as optim
from torch.nn import ModuleList
import torch.nn.functional as F

from discourse.predictors import DiscourseClassifierPredictor
from discourse.dataset_readers import ClaimAnnotationReaderCSV
from discourse.models import DiscourseClassifier

from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor
from allennlp.common import Params
from allennlp.common.file_utils import cached_path
from allennlp.common.util import JsonDict

from allennlp.data.fields import TextField, LabelField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.vocabulary import Vocabulary

from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.iterators import BucketIterator
from allennlp.training.trainer import Trainer
from allennlp.training.learning_rate_schedulers import LearningRateScheduler



EMBEDDING_DIM = 300
DISCOURSE_MODEL_PATH = 'https://s3-us-west-2.amazonaws.com/pubmed-rct/model.tar.gz'
TRAIN_PATH = 'https://s3-us-west-2.amazonaws.com/pubmed-rct/train_labels.csv'
VALIDATION_PATH = 'https://s3-us-west-2.amazonaws.com/pubmed-rct/validation_labels.csv'
archive = load_archive(DISCOURSE_MODEL_PATH)
predictor = Predictor.from_archive(archive, 'discourse_predictor')


def flatten_dataset(df):
    """
    Flatten Gold Standard JSON data for Claim Extraction
    """
    sentence_data = []
    for _, r in df.iterrows():
        sentence_data.extend(list(zip(r['sentences'], r['labels'])))
    flatten_df = pd.DataFrame(sentence_data, columns=['sentence', 'label'])
    return flatten_df


class ClaimClassifierPredictor(Predictor):
    """"
    Predictor wrapper for the AcademicPaperClassifier
    """
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        sentence = json_dict['sentence']
        instance = self._dataset_reader.text_to_instance(sentence=sentence)
        return instance


if __name__ == '__main__':
    model = predictor._model
    # freeze all parameters
    for param in list(model.parameters()):
        param.requires_grad = False
    model.classifier_feedforward._linear_layers = ModuleList([torch.nn.Linear(2 * EMBEDDING_DIM, EMBEDDING_DIM), 
                                                              torch.nn.Linear(EMBEDDING_DIM, 2)])
    vocab = predictor._model.vocab
    vocab._token_to_index['labels'] = {'0': 0, '1': 1}
    # freeze all layers except top layer
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    iterator = BucketIterator(batch_size=64, 
                              sorting_keys=[("sentence", "num_tokens")])
    iterator.index_with(vocab)

    reader = ClaimAnnotationReaderCSV()
    train_dataset = reader.read(cached_path(TRAIN_PATH))
    validation_dataset = reader.read(cached_path(VALIDATION_PATH))

    # train unfreeze top layers
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        iterator=iterator,
        validation_iterator=iterator,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        patience=5,
        num_epochs=100, 
        cuda_device=-1
    )
    trainer.train()

    # unfreeze all layers and train
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        iterator=iterator,
        validation_iterator=iterator,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        patience=5,
        num_epochs=200, 
        cuda_device=-1
    )
    for param in list(model.parameters())[1:]:
        param.requires_grad = True
    trainer.train()

    claim_predictor = ClaimClassifierPredictor(model, dataset_reader=reader)
    validation_df = pd.read_csv(cached_path(VALIDATION_PATH))
    validation_df['class_probabilities'] = validation_df.sentence.map(lambda x: claim_predictor.predict_json({'sentence': x})['class_probabilities'])
    validation_df['predicted_label'] = validation_df.class_probabilities.map(lambda x: np.argmax(x))
    y_true, y_pred = validation_df.label.astype(int).values, validation_df.predicted_label.astype(int).values
    print(precision_recall_fscore_support(y_true, y_pred, average='binary'))

    # see the optimal threshold for validation set
    p_claim = np.vstack(validation_df.class_probabilities.map(np.array).values)
    for threshold in np.arange(0.25, 0.7, 0.02):
        y_pred = (p_claim[:, 1] >= threshold)
        print('Threshold = {}, F1-score = {}'.format(threshold, f1_score(y_true, y_pred)))