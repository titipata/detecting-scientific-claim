"""
Run with `discourse` folder,
"""
from typing import Iterator, List, Dict, Optional
import pandas as pd
from itertools import chain
from sklearn.model_selection import train_test_split

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

from allennlp.data.fields import TextField, LabelField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.vocabulary import Vocabulary

from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.iterators import BucketIterator
from allennlp.training.trainer import Trainer


class ClaimDatasetReader(DatasetReader):
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


if __name__ == '__main__':

    DISCOURSE_MODEL_PATH = 'https://s3-us-west-2.amazonaws.com/pubmed-rct/model.tar.gz'
    SAMPLE_TRAINING_PATH = 'https://s3-us-west-2.amazonaws.com/pubmed-rct/sample_training.csv'
    SAMPLE_VALIDATION_PATH = 'https://s3-us-west-2.amazonaws.com/pubmed-rct/sample_validation.csv'
    archive = load_archive(DISCOURSE_MODEL_PATH) # discourse model
    predictor = Predictor.from_archive(archive, 'discourse_classifier')

    model = predictor._model
    model.classifier_feedforward._linear_layers = ModuleList([torch.nn.Linear(600, 300), torch.nn.Linear(300, 2)])
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    iterator = BucketIterator(batch_size=64, 
                          sorting_keys=[("sentence", "num_tokens")])
    iterator.index_with(predictor._model.vocab)

    reader = ClaimDatasetReader()
    train_dataset = reader.read(cached_path(SAMPLE_TRAINING_PATH))
    validation_dataset = reader.read(cached_path(SAMPLE_VALIDATION_PATH))

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        iterator=iterator,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        patience=2,
        num_epochs=5, 
        cuda_device=-1
    )
    trainer.train()
