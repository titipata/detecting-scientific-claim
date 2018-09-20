"""
Snippet to prepare the dataset for annotation

requirement: https://github.com/titipata/pubmed_parser
"""
import os
from collections import Counter
from glob import glob
import pandas as pd
from utils import save_json, read_json
import pubmed_parser as pp
import spacy


nlp = spacy.load('en_core_web_sm')
STRUCTURE_ABSTRACT = ['INTRODUCTION', 'BACKGROUND', 'METHOD', 'RESULT', 'OBJECTIVE']

def parse_medline_articles(path='medline', 
                           saved_path='parsed_articles', 
                           year_start=2000, 
                           year_stop=2018):
    """
    Give a ``path`` to folder locating .xml.gz of Medline articles, 
    parse and save the parsed articles to ``saved_path`` 

    Input
    =====
    path: str, path to folder with all .xml.gz files
    saved_path: str, path to saved articles
    year_start: int, first year to save parsed article
    year_stop: int, last year to save parsed article
    """
    paths = glob(os.path.join(path, '*.xml.gz'))

    # check if the directory is not there
    if not os.path.isdir(saved_path):
        os.mkdir(saved_path)

    for i, path in enumerate(paths):
        all_parsed_papers = []
        parsed_papers = pp.parse_medline_xml(path)
        for paper in parsed_papers:
            try:
                if int(paper['pubdate']) >= year_start and int(paper['pubdate']) <= year_stop:
                    all_parsed_papers.append(paper)
            except:
                pass
        save_json(all_parsed_papers, os.path.join(saved_path, 'parsed_%d.json' % i))
    print('done!')


def combine_parsed_medline_articles(saved_path='parsed_articles', 
                                    year_start=2010, 
                                    year_end=2019):
    """
    Give a path to folder locating JSON files, 
    return all parsed paper

    Input
    =====
    saved_path: str, path to saved JSON folder

    Output
    ======
    parsed_papers: list, list of all parsed papers 
    """
    paths = glob(os.path.join(saved_path, '*.json'))

    parsed_papers = []
    for path in paths:
        papers = read_json(path)
        papers = [paper for paper in papers if int(paper['pubdate']) in range(2010, 2019)]
        parsed_papers.extend(papers)
    return parsed_papers


def calculate_journal_stats(parsed_papers, n_journal=100, save_figure=False):
    """
    Calculate journal stats from parsed articles

    Input
    =====
    parsed_papers: list, list of parsed articles

    Output
    ======
    journals_df: DataFrame, pandas dataframe consisting of journal and number of articles from a given journal
    """
    journals = [paper['journal'] for paper in parsed_papers]
    journals_df = pd.DataFrame(list(Counter(journals).items()), columns=['journal', 'number_of_papers'])
    journals_df.sort_values('number_of_papers', ascending=False).head(n_journal)

    if save_figure:
        import matplotlib.pyplot as plt
        r = list(range(len(journals_df)))
        ax = plt.gca()
        ax.scatter(r, journals_df['number_of_papers'])
        ax.set_xticks(r)
        ax.set_xticklabels(journals_df['journal'], rotation=90)
        ax.set_yscale('log')
        ax.set_ylabel('Number of Publications', fontsize=20)
        ax.set_title('Publication distribution of MEDLINE articles', fontsize=20)
        plt.savefig('journal_count.svg')
        plt.show()

    return journals_df


def is_structured_abstract(abstract):
    """
    Check if the given abstract is structured abstract or not
    """
    if any([s in abstract for s in STRUCTURE_ABSTRACT]):
        return True
    else:
        return False


def calculate_n_sents(abstract):
    """
    Calculate number of sentences in a given abstract
    """
    abstract = nlp(abstract)
    return len([sent for sent in abstract.sents])
    

def sample_articles(parsed_papers, 
                    n_sample=3000, 
                    random_state=1302, 
                    n_sents_max=15, 
                    n_sents_min=5):
    """
    Given a list of articles, sample articles for annotation task
    """
    article_df = pd.DataFrame(parsed_papers)
    article_df.dropna(axis=0, subset=['abstract', 'pmid'], inplace=True)
    sample_abstract_df = article_df.sample(n=n_sample, random_state=random_state)
    sample_abstract_df['is_structured_abstract'] = sample_abstract_df['abstract'].map(is_structured_abstract)
    sample_abstract_df = sample_abstract_df[~sample_abstract_df.is_structured_abstract]
    sample_abstract_df['pmid'] = sample_abstract_df['pmid'].map(int)
    sample_abstract_df['n_sents'] = sample_abstract_df['abstract'].map(calculate_n_sents)
    sample_abstract_df = sample_abstract_df[(sample_abstract_df.n_sents >= n_sents_min) & (sample_abstract_df.n_sents <= n_sents_max)]
    return sample_abstract_df