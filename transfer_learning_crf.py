from typing import Iterator, List, Dict, Optional
import os
import json
import numpy as np
import pandas as pd
from itertools import chain
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from allennlp.common.util import JsonDict

import torch
import torch.optim as optim
from torch.nn import ModuleList
import torch.nn.functional as F

from discourse.predictors import DiscourseClassifierPredictor
from discourse.dataset_readers import PubmedRCTReader
from discourse.models import DiscourseClassifier

from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor
from allennlp.common.file_utils import cached_path

from allennlp.data.fields import Field, TextField, LabelField, ListField, SequenceLabelField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.vocabulary import Vocabulary

from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.iterators import BucketIterator, BasicIterator
from allennlp.training.trainer import Trainer

from allennlp.modules import Seq2VecEncoder, TimeDistributed, TextFieldEmbedder, ConditionalRandomField, FeedForward
from torch.nn.modules.linear import Linear


def save_json(ls, file_path):
    """
    Save list of dictionary to JSON
    """
    with open(file_path, 'w') as fp:
        fp.write('\n'.join(json.dumps(i) for i in ls))


class CrfClaimReader(DatasetReader):
    """
    Reads claim annotation dataset in JSON format
    
    Parameters
    ----------
    tokenizer : ``Tokenizer``, optional (default=``WordTokenizer()``)
        We use this ``Tokenizer`` for both the premise and the hypothesis.  See :class:`Tokenizer`.
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We similarly use this for both the premise and the hypothesis.  See :class:`TokenIndexer`.
    """

    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

    def _read(self, file_path):
        file_path = cached_path(file_path)
        with open(file_path, 'r') as file:
            for line in file:
                example = json.loads(line)
                sents = example['sentences']
                labels = [str(label) for label in example['labels']]
                yield self.text_to_instance(sents, labels)

    def text_to_instance(self,
                         sents: List[str],
                         labels: List[str] = None) -> Instance:
        fields: Dict[str, Field] = {}
        tokenized_sents = [self._tokenizer.tokenize(sent) for sent in sents]
        sentence_sequence = ListField([TextField(tk, self._token_indexers) for tk in tokenized_sents])
        fields['sentences'] = sentence_sequence
        if labels is not None:
            fields['labels'] = SequenceLabelField(labels, sentence_sequence)
        return Instance(fields)


class ClaimCrfPredictor(Predictor):
    """"
    Predictor wrapper for the AcademicPaperClassifier
    """
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        sentences = json_dict['sentences']
        instance = self._dataset_reader.text_to_instance(sents=sentences)
        return instance


if __name__ == '__main__':
    SAMPLE_TRAINING_PATH = 'https://s3-us-west-2.amazonaws.com/pubmed-rct/sample_training.jsonl'
    SAMPLE_VALIDATION_PATH = 'https://s3-us-west-2.amazonaws.com/pubmed-rct/sample_validation.jsonl'
    DISCOURSE_MODEL_PATH = 'https://s3-us-west-2.amazonaws.com/pubmed-rct/model_crf.tar.gz'
    archive = load_archive(DISCOURSE_MODEL_PATH)
    discourse_predictor = Predictor.from_archive(archive, 'discourse_crf_classifier')

    model = discourse_predictor._model
    num_classes, constraints, include_start_end_transitions = 2, None, False
    model.classifier_feedforward._linear_layers = ModuleList([torch.nn.Linear(400, 200), torch.nn.Linear(200, 2)])
    model.crf = ConditionalRandomField(num_classes, constraints, 
                                    include_start_end_transitions=include_start_end_transitions)
    model.label_projection_layer = TimeDistributed(Linear(400, num_classes))

    reader = CrfClaimReader()
    train_dataset = reader.read(SAMPLE_TRAINING_PATH)
    validation_dataset = reader.read(SAMPLE_VALIDATION_PATH)
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
        patience=5,
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
        patience=5,
        num_epochs=100,
        cuda_device=-1
    )
    trainer.train()

    # make prediction
    claim_predictor = ClaimCrfPredictor(model, dataset_reader=reader)
    y_pred, y_true = [], []
    for val in validation_dataset:
        pred = claim_predictor.predict_instance(val)
        logits = torch.FloatTensor(pred['logits'])
        best_paths = model.crf.viterbi_tags(torch.FloatTensor(pred['logits']).unsqueeze(1), 
                                            torch.LongTensor(pred['mask']))
        predicted_labels = [x[0] for x, y in best_paths]
        y_pred.extend(predicted_labels)
        y_true.extend([int(l) for l in val['labels']])
    print(f1_score(y_true, y_pred))