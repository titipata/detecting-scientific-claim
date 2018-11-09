from typing import List
import os
import sys
import json
import pandas as pd
from itertools import chain
from lxml import etree, html
from urllib.request import urlopen
from sklearn.metrics import cohen_kappa_score


PUBMED_URL = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&retmode=xml&id=%s'


def stringify_children(node):
    """
    Filters and removes possible Nones in texts and tails
    ref: http://stackoverflow.com/questions/4624062/get-all-text-inside-a-tag-in-lxml
    """
    parts = ([node.text] +
             list(chain(*([c.text, c.tail] for c in node.getchildren()))) +
             [node.tail])
    return ''.join(filter(None, parts))


def parse_pubmed_xml(pmid):
    """
    Parse article information for the given PMID
    """
    pmid = str(pmid)
    url = PUBMED_URL % pmid
    page = urlopen(url).read()
    tree = html.fromstring(page)
    abstract = ''
    for e in tree.xpath('//abstract/abstracttext'):
        if e is not None:
            abstract += stringify_children(e).strip()
    title = ' '.join([e.text for e in tree.xpath('//articletitle')
                     if e is not None])
    return {'title': title, 'abstract': abstract}


def read_json(file_path):
    """
    Read collected file from path
    """
    if not os.path.exists(file_path):
        return []
    else:
        with open(file_path, 'r') as fp:
            ls = [json.loads(line) for line in fp]
        return ls


def save_json(ls, file_path):
    """
    Save list of dictionary to JSON
    """
    with open(file_path, 'w') as fp:
        fp.write('\n'.join(json.dumps(i) for i in ls))


def calculate_cohen_kappa_score(df: pd.DataFrame, users: List = []):
    """
    Calculage inter-annotation agreement given the labels dataframe

    dataframe columns include "user_id", "paper_id", "labels" (List)

    >> calculate_cohen_kappa_score(df, users=['user1', 'user2'])
    """
    assert(len(users) == 2)
    annotation_sel_df = df[df['user_id'].map(lambda x: x in users)]
    size_df = annotation_sel_df.groupby('paper_id').size().reset_index()
    annotation_sel_df = annotation_sel_df.merge(size_df[size_df[0] == 2][['paper_id']])

    annotations_1, annotations_2 = [], []
    for _, df in annotation_sel_df.groupby('paper_id'):
        annotations_1.extend(df.labels.iloc[0])
        annotations_2.extend(df.labels.iloc[1])

    return cohen_kappa_score(annotations_1, annotations_2)