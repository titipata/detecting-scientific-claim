"""
Snippet to prepare the dataset for annotation

requirement: https://github.com/titipata/pubmed_parser
"""
import os
from glob import glob
from utils import save_json, read_json
import pubmed_parser as pp


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


def combine_parsed_medline_articles(saved_path='parsed_articles'):
    """
    Give a path to folder locating JSON files, return 
    """
    paths = glob(os.path.join(saved_path, '*.json'))

    parsed_papers = []
    for path in paths:
        parsed_papers.extend(read_json(path))
    return parsed_papers

    