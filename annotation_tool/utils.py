from itertools import chain
from lxml import etree, html
from urllib.request import urlopen


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
