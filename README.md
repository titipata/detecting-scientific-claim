# Claim Extraction for Scientific Publications

Detecting claim from scientific publication using discourse and sentence embedding.


## Download

You can preprocess [Pubmed RCT](https://github.com/Franck-Dernoncourt/pubmed-rct) file
using `spacy` by running `python preprocess.py`
(note that you have to put `train.txt`, `test.txt` in `pubmed-rct/PubMed_200k_RCT/`)
or alternatively download JSON directly from Amazon S3.

```bash
wget https://s3-us-west-2.amazonaws.com/pubmed-rct/train.json -P pubmed-rct/PubMed_200k_RCT/
wget https://s3-us-west-2.amazonaws.com/pubmed-rct/test.json -P pubmed-rct/PubMed_200k_RCT/
```


## Requirements

- [Pubmed RCT](https://github.com/Franck-Dernoncourt/pubmed-rct) - dataset
- [AllenNLP](https://github.com/allenai/allennlp)
- [spacy](https://github.com/explosion/spaCy)
