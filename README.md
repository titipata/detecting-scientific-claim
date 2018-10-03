# Claim Extraction for Scientific Publications

Detecting claim from scientific publication using discourse and sentence embedding 
powered by [AllenNLP](https://github.com/allenai/allennlp).


## Training discourse model

Running AllenNLP to train a discourse model as follows

```bash
allennlp train experiments/pubmed_rct.json -s output --include-package discourse
```

We point data location to Amazon S3 directly in `pubmed_rct.json`
so you do not need to download the data locally. Change `cuda_device` to `-1` in `pubmed_rct.json`
if you want to run on CPU. There are more experiments available in `experiments` folder.

**Note** that you have to remove `output` folder first before running.


## Predicting discourse

We trained the Bidirectional LSTM model on structured abstracts from Pubmed to predict
discourse probability (`RESULTS`, `METHODS`, `CONCLUSIONS`, `BACKGROUND`, `OBJECTIVE`)
of a given sentence. You can download trained model from Amazon S3

```bash
wget https://s3-us-west-2.amazonaws.com/pubmed-rct/model.tar.gz
```

and run web service for discourse prediction task as follow

```bash
bash web_service.sh
```

To test the train model with provided examples [`fixtures.json`](pubmed-rct/PubMed_200k_RCT/fixtures.json),
simply run the following to predict labels.


```bash
allennlp predict output/model.tar.gz \
    pubmed-rct/PubMed_200k_RCT/fixtures.json \
    --include-package discourse \
    --predictor discourse_classifier
```

To evaluate discourse model, you can run the following command

```bash
allennlp evaluate output/model.tar.gz \
  --evaluation-data-file https://s3-us-west-2.amazonaws.com/pubmed-rct/test.json \
  --include-package discourse
```


## Predicting claim (web service)

First, download fastText pre-trained word vector

```bash
wget https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.en.zip
```

then unzip it, and put `wiki.en.bin` in this folder. Now, you can run `flask`
application to detect claims as follows

```bash
export FLASK_APP=main.py
flask run --host=0.0.0.0 # this will serve at port 5000
```

The interface will look something like this

<p float="left">
  <img src="static/interface.png" width="600" />
</p>

And output will look something like the following (highlight means claim,
  tag behind the sentence is discourse prediction)

<p float="left">
  <img src="static/output.png" width="600" />
</p>


We will release a dataset and model of scientific claims tagged by expert soon
(in a coming months).


## Requirements

- [Python 3.6](https://www.python.org/downloads/release/python-360/)
- [AllenNLP](https://github.com/allenai/allennlp) >= 0.6.1
- [spacy](https://github.com/explosion/spaCy)
- [fastText](https://github.com/facebookresearch/fastText)
- [Pubmed RCT](https://github.com/Franck-Dernoncourt/pubmed-rct) - dataset


## Citing the repository

We do not have the publication related to the repository yet. You can cite the
Github repository for now:


Titipat Achakulvisut, Chandra Bhagavatula, Daniel E Acuna, Konrad P Kording, _Claim Extraction for Scientific Publications_,
https://github.com/titipata/detecting-scientific-claim, 2018

or using BibTeX

```
@misc{Achakulvisut2018,
  author = {Achakulvisut T, Bhagavatula C, Acuna DE, Kording KP},
  title = {Claim Extraction for Scientific Publications},
  year = {2018},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/titipata/detecting-scientific-claim}}
}
```

## Acknowledgement

This project is done at the [Allen Institute for Artificial Intelligence](https://allenai.org/)
and [Konrad Kording lab, University of Pennsylvania](http://kordinglab.com/)
