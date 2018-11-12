# run with ``discourse`` folder
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


def pad(l, size, padding=0):
    return l + [padding] * abs((len(l)-size))


if __name__ == '__main__':
    # load trained discourse
    DISCOURSE_MODEL_PATH = 'https://s3-us-west-2.amazonaws.com/pubmed-rct/model.tar.gz'
    SAMPLE_TRAINING_PATH = 'https://s3-us-west-2.amazonaws.com/pubmed-rct/sample_training.csv'
    archive = load_archive(DISCOURSE_MODEL_PATH) # discourse model
    predictor = Predictor.from_archive(archive, 'discourse_classifier')

    # get model, vocab
    model = predictor._model
    model.classifier_feedforward._linear_layers = ModuleList([torch.nn.Linear(600, 300), torch.nn.Linear(300, 2)])
    vocab = predictor._model.vocab

    # freeze all layers except top layer
    for param in list(model.parameters())[:-4]:
        param.requires_grad = False
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    n_epochs = 100
    batch_size = 64
    tokenizer = WordTokenizer()
    single_id_tokenizer = SingleIdTokenIndexer()

    for epoch in range(n_epochs):
        training = pd.read_csv(cached_path(SAMPLE_TRAINING_PATH), chunksize=batch_size)
        for df in training:
            optimizer.zero_grad()
            token_indices = []
            for sentence in list(df['sentence']):
                # transform to long tensor
                token_indice = single_id_tokenizer.tokens_to_indices(tokenizer.tokenize(sentence), 
                                                                    vocabulary=vocab, 
                                                                    index_name='tokens')['tokens']
                token_indices.append(token_indice)
            max_token_length = max(list(map(len, token_indices)))
            token_indices = [pad(token_indice, max_token_length) for token_indice in token_indices]
            sentences = torch.LongTensor(token_indices)
            labels = torch.LongTensor(df['label'].values)
            y_pred = F.softmax(model({'tokens': sentences})['logits'], dim=-1)
            loss = model.loss(y_pred, labels)
            loss.backward()
            optimizer.step()
        print(float(loss))