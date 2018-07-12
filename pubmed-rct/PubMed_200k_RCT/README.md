# Pubmed 200k RCT

placeholder for Pubmed RCT dataset

## Download Dataset

You can preprocess [Pubmed RCT](https://github.com/Franck-Dernoncourt/pubmed-rct) file
using `spacy` by running `python preprocess.py`
(note that you have to put `train.txt`, `dev.txt`, `test.txt` in `pubmed-rct/PubMed_200k_RCT/`)
or alternatively download JSON directly from Amazon S3.

```bash
wget https://s3-us-west-2.amazonaws.com/pubmed-rct/train.json
wget https://s3-us-west-2.amazonaws.com/pubmed-rct/dev.json
wget https://s3-us-west-2.amazonaws.com/pubmed-rct/test.json
```

You do not have to download the dataset locally for discourse training
since we point to Amazon S3 directly.
