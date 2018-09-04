import json

from typing import Dict
from overrides import overrides

from allennlp.common import Params
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, LabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer


@DatasetReader.register("pubmed_rct")
class PubmedRCTReader(DatasetReader):
    """
    Reads a file from Pubmed RCT dataset. We preprocess Pubmed RCT to JSON format
    where each line is a dictionary of
        - `label` - label of the sentence
        - `sentence` - list of tokenized words of a sentence
        - `pos` - list of tokenized part-of-speech tags
        - `sentence_text` - full text of the sentence

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

    @overrides
    def _read(self, file_path):
        file_path = cached_path(file_path)
        with open(file_path, 'r') as file:
            for line in file:
                example = json.loads(line)
                label = example['label']
                sent = example['sentence_text']
                yield self.text_to_instance(sent, label)

    @overrides
    def text_to_instance(self,
                         sent: str,
                         label: str = None) -> Instance:
        fields: Dict[str, Field] = {}
        tokenized_sent = self._tokenizer.tokenize(sent)
        fields['sentence'] = TextField(tokenized_sent, self._token_indexers)
        if label is not None:
            fields['label'] = LabelField(label)
        return Instance(fields)
