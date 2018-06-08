# Claim Extraction for Scientific Publications

Detecting claim from scientific publication using discourse and sentence embedding.


## Download Dataset

You can preprocess [Pubmed RCT](https://github.com/Franck-Dernoncourt/pubmed-rct) file
using `spacy` by running `python preprocess.py`
(note that you have to put `train.txt`, `test.txt` in `pubmed-rct/PubMed_200k_RCT/`)
or alternatively download JSON directly from Amazon S3.

```bash
wget https://s3-us-west-2.amazonaws.com/pubmed-rct/train.json -P pubmed-rct/PubMed_200k_RCT/
wget https://s3-us-west-2.amazonaws.com/pubmed-rct/test.json -P pubmed-rct/PubMed_200k_RCT/
```

The dataset is not necessary for discourse training since we point to Amazon S3 directly.


## Training discourse model

Running AllenNLP to train discourse model as follows

```bash
python -m allennlp.run train pubmed_rct.json -s output --include-package discourse
```

**Note** that you have to remove `output` folder first before running. `pubmed_rct.json`
contains parameters for the model, change `cuda_device` to `0` if you want to run on GPU.


## Predicting claim

We will release a dataset of scientific claims tagged by expert soon (in few months).


## Requirements

- [Pubmed RCT](https://github.com/Franck-Dernoncourt/pubmed-rct) - dataset
- [AllenNLP](https://github.com/allenai/allennlp)
- [spacy](https://github.com/explosion/spaCy)
