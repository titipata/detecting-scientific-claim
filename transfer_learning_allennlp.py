"""
Transfer learning for claim prediction. Run this script with `discourse` folder,
"""
from typing import Iterator, List, Dict, Optional
import numpy as np
import pandas as pd
from itertools import chain
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

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
from allennlp.common.util import JsonDict

from allennlp.data.fields import TextField, LabelField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.vocabulary import Vocabulary

from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.iterators import BucketIterator
from allennlp.training.trainer import Trainer


class ClaimDatasetReader(DatasetReader):
    """
    Sentence reader for Claim dataset
    """
    def __init__(self, 
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None, 
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        
    def _read(self, file_path: str) -> Iterator[Instance]:
        reader = pd.read_csv(file_path, chunksize=1)
        for row in reader:
            d = dict(row.iloc[0])
            sentence = d['sentence']
            label = str(d['label'])
            yield self.text_to_instance(sentence, label)
    
    def text_to_instance(self, 
                         sentence: str, 
                         label: str=None) -> Instance:
        """
        Turn title, abstract, and venue to instance
        """
        tokenized_sentence = self._tokenizer.tokenize(sentence)
        sentence_field = TextField(tokenized_sentence, self._token_indexers)
        fields = {'sentence': sentence_field}
        if label is not None:
            fields['label'] = LabelField(label)
        return Instance(fields)


class ClaimClassifierPredictor(Predictor):
    """"
    Predictor wrapper for the claim prediction task
    """
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        sentence = json_dict['sentence']
        instance = self._dataset_reader.text_to_instance(sentence=sentence)
        return instance


if __name__ == '__main__':
    """
    Download pretrained discourse model and use it to train claim prediction model
    """
    DISCOURSE_MODEL_PATH = 'https://s3-us-west-2.amazonaws.com/pubmed-rct/model.tar.gz'
    SAMPLE_TRAINING_PATH = 'https://s3-us-west-2.amazonaws.com/pubmed-rct/sample_training.csv'
    SAMPLE_VALIDATION_PATH = 'https://s3-us-west-2.amazonaws.com/pubmed-rct/sample_validation.csv'
    archive = load_archive(DISCOURSE_MODEL_PATH) # discourse model
    predictor = Predictor.from_archive(archive, 'discourse_classifier')

    model = predictor._model
    model.classifier_feedforward._linear_layers = ModuleList([torch.nn.Linear(600, 300), torch.nn.Linear(300, 2)])
    vocab = predictor._model.vocab
    vocab._token_to_index['labels'] = {'0': 0, '1': 1}
    # freeze all layers except top layer
    for param in list(model.parameters())[:-4]:
        param.requires_grad = False
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    iterator = BucketIterator(batch_size=64, 
                              sorting_keys=[("sentence", "num_tokens")])
    iterator.index_with(vocab)

    reader = ClaimDatasetReader()
    train_dataset = reader.read(cached_path(SAMPLE_TRAINING_PATH))
    validation_dataset = reader.read(cached_path(SAMPLE_VALIDATION_PATH))

    # unfreeze top layers and train
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
        num_epochs=87, 
        cuda_device=-1
    )
    trainer.train()

    # print out validation F-score
    claim_predictor = ClaimClassifierPredictor(model, dataset_reader=reader)
    validation_df = pd.read_csv(cached_path(SAMPLE_VALIDATION_PATH))
    validation_df['class_probabilities'] = validation_df.sentence.map(lambda x: claim_predictor.predict_json({'sentence': x})['class_probabilities'])
    validation_df['predicted_label'] = validation_df.class_probabilities.map(lambda x: np.argmax(x))
    print(f1_score(validation_df.label, validation_df.predicted_label))