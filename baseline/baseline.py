import os
import re
import spacy
from glob import glob


nlp = spacy.load('en_core_web_sm')
keyword_path = os.path.join(os.getcwd(), 'keywords')  # path to keywords folder

TRIGGER_WORDS = frozenset(
    [
        'project', 'framework', 'methodology', 'algorithm', 'approach', 'system', 'platform',
        'prototype', 'work', 'paper', 'project', 'article', 'research', 'study', 'prototype',
        'library', 'finding'
    ]
)


def load_keywords(path=keyword_path):
    """
    Load list of 186 keywords listed in  from the given path
    """
    keyword_files = glob(os.path.join(path, '*.txt'))
    keywords = []
    for keyword_file in keyword_files:
        if not any([d in keyword_file for d in ['deictic_cliche', 'deictic_phrase']]):
            with open(keyword_file, 'r') as f:
                keywords.extend([w.strip() for w in f.readlines()])
    keywords = frozenset(keywords)
    return keywords


def find_noun_phrase_offset(token):
    """
    Given tokenized sentence string,
    return start/end position of noun chunks token
    (end in this case is one position after noun phrase token)
    """
    noun_phrase_position = [(s.start, s.end) for s in token.noun_chunks]
    return noun_phrase_position


def find_deictic(token, keywords=[]):
    """
    Give a tokenized sentence input (and list of keywords),
    return all start and end indices for deictic
    """
    deilectics = []
    n_token = len(token)
    token_zip = list(zip(token, token[1:]))

    for token1, token2 in token_zip:
        # deilectic rule 1: This paper presents a use case of ...
        if (token1.pos_ == 'DET') and any([token2.lemma_ in g for g in keywords]):
            deilectics.append((token1.i, token2.i))
        # deilectic rule 2: Here, we demonstrate how our interpretation ...
        if token1.pos_ == 'ADV' and token2.pos_ == 'PUNCT' and token1.lemma_ not in ['so']:
            deilectics.append((token1.i, token2.i))
    # deilectic rule 3: Our approach is compatible with the ...
    noun_phrase_position = find_noun_phrase_offset(token)
    if noun_phrase_position:
        for (start, end) in noun_phrase_position:
            tag_list = [t.text.lower() for t in token[start:end]]
            is_we_in = any([w in tag_list for w in ['we', 'our', 'paper']])
            if token[min(end, n_token - 1)].pos_ == 'VERB' and is_we_in:
                deilectics.append((start, end - 1))
    deilectics = list(set(deilectics))
    return deilectics


def find_meta_discourse(token, keywords=[]):
    """
    Give a tokenized sentence input,
    return all start and end indices for meta discourse
    """
    meta_discourses = []
    deilectics = find_deictic(token, keywords)
    n_token = len(token)
    token_zip = list(zip(token, token[1:]))
    for (start, end) in deilectics:
        n_sel = end
        for j in range(end, min(end + 3, n_token)):
            if token[j].pos_ == 'NOUN':
                n_sel = j
        # select two tokens ahead
        token1 = token[min(n_sel + 1, n_token - 1)]
        token2 = token[min(n_sel + 2, n_token - 1)]
        # metadiscourse rule 1: deictic + verb_presentation
        if token1.pos_ == 'VERB' and (token1.lemma_ in keywords):
            meta_discourses.append((start, n_sel + 1))
        # metadiscourse rule 2: deictic + pronoun + verb_presentation
        if token1.pos_ == 'PRON' and (token2.lemma_ in keywords):
            meta_discourses.append((start, n_sel + 2))

    # metadiscourse rule 3: pron + verb_presentation, We built the first ...
    for token1, token2 in token_zip:
        if (token1.pos_ == 'PRON') and (token2.pos_ == 'VERB') and (token2.pos_ in keywords):
            meta_discourses.append((token1.i, token2.i))
    return meta_discourses


def find_contribution(token, keywords=[]):
    """
    Give a tokenized sentence input,
    return all start and end indices for contribution statement
    """
    contributions = []
    n_token = len(token)
    meta_discourses = find_meta_discourse(token, keywords)
    skip_noun_phrase_dict = dict(find_noun_phrase_offset(token))

    # contribution rule 1: metadiscourse + noun phrase
    for (start, end) in meta_discourses:
        if (skip_noun_phrase_dict.get(end + 1) is
            not None) or (token[min(end + 1, n_token - 1)].pos_ == 'NOUN'):
            contributions.append((start, skip_noun_phrase_dict.get(end + 1)))
            
    # contribution rule 2: metadiscourse + adverb + noun phrase
    for (start, end) in meta_discourses:
        if (token[min(end + 1, n_token - 1)].pos_ == 'ADV' and token[min(end + 2, n_token - 1)].pos_ == 'ADJ') or \
           (token[min(end + 1, n_token - 1)].pos_ == 'ADV' and token[min(end + 2, n_token - 1)].pos_ == 'NOUN'):
            contributions.append((start, end + 2))
    return contributions


def find_claim(token, keywords=set()):
    """
    Give a tokenized sentence input from spacy,
    return all start and end indices of claim statement

    Parameters
    ==========
    token: spacy token of the string 
    keywords: set of keywords, default to all keywords located in 'keywords' folder

    Output
    ======
    claims: list of tuple, list of tuple which contains token position of claims, 
        if list is not empty meaning that there is claim in the given token

    Example
    =======
        keywords = load_keywords('keywords')
        find_claim(nlp('In this study ...'), keywords) # return position of claim, if 
    """
    claims = []
    if len(token) < 5 or not any(
        [t.text.lower() in TRIGGER_WORDS.union({'we', 'our'}) for t in token]
    ):
        return claims
    deilectics = find_deictic(token, keywords)
    meta_discourses = find_meta_discourse(token, keywords)
    skip_noun_phrase_dict = dict(find_noun_phrase_offset(token))
    n_token = len(token)
    lemma_all = [t.lemma_ for t in token]
    lemma_in_keywords = any([(l in keywords) for l in lemma_all])

    # claim rule 1: meta discourse + det + adj + trigger
    # We built the first BauDenkMalNetz prototype using SMW
    for (start, end) in meta_discourses:
        token1 = token[min(end + 1, n_token - 1)]
        token2 = token[min(end + 2, n_token - 1)]
        if (token1.pos_ == 'DET'
           ) and (token2.pos_ == 'ADJ' or token2.pos_ == 'ADV') and lemma_in_keywords:
            claims.append((start, end + 2))

    for (start, end) in deilectics:
        # claim rule 2: deictic + adjective or adverb
        token1 = token[min(end + 1, n_token - 1)]
        token2 = token[min(end + 2, n_token - 1)]
        token3 = token[min(end + 3, n_token - 1)]
        if token1.pos_ == 'VERB' and (token2.pos_ == 'ADJ' or token2.pos_ == 'ADV') \
           and 'we ' in ' '.join([t.text.lower() for t in token[start: min(end + 2, n_token - 1)]]):
            claims.append((start, end + 2))

        # claim rule 3: deilectics + VBP + ..., We have found that ...
        # This paper has presented a computational strategy for ...
        is_kw_in = any([t.text.lower() in TRIGGER_WORDS for t in token[start:end]])
        if (
            token1.pos_ == 'VERB' and token2.pos_ == 'VERB' and
            (token3.pos_ in ['ADP', 'DET', 'ADJ', 'NUM'])
        ) and is_kw_in:
            claims.append((start, end + 3))

        # claim adding rule: Our system maintains a set ...
        deilectics_text = ' '.join([t.text for t in token[start:end + 1]])
        if (token1.pos_ == 'VERB') and ('our' in deilectics_text.lower()):
            claims.append((start, end + 1))

        # claim rule 5: This work is an important first step ...
        if token1.pos_ == 'VERB' and (
            (token2.pos_ == 'DET' and token3.pos_ == 'ADJ') or token2.pos_ == 'ADJ'
        ):
            claims.append((start, end + 2))

        if skip_noun_phrase_dict.get(token1.i + 1) is not None and is_kw_in:
            claims.append((start, skip_noun_phrase_dict.get(token1.i + 1)))

        # Finally, we point out how to use the FOAF ...
        if (token1.pos_ == 'PRON' and token1.text.lower() == 'we') and token2.pos_ == 'VERB':
            claims.append((start, end))

    # claim rule 4: Our study also shows...
    noun_phrase_position = find_noun_phrase_offset(token)
    for (start, end) in noun_phrase_position:
        token1 = token[min(end, n_token - 1)]
        token2 = token[min(end + 1, n_token - 1)]
        tag_list = [t.text.lower() for t in token[start:end]]
        is_we_in = any([w in tag_list for w in ['we', 'our', 'paper', 'study']])
        if (token1.lemma_ in keywords) and is_we_in:  # or (lemma2 in keywords)
            claims.append((start, end))
        if token1.pos_ == 'ADV' and token2.lemma_ in keywords:
            claims.append((start, end + 1))
        # example: In this paper, we discuss a web-first approach
        if token1.pos_ == 'PUNCT' and (token2.pos_ == 'PRON' and token2.text.lower() == 'we'):
            claims.append((start, end + 1))
        # example: In this manuscript we produce and analyze ...
        if token1.pos_ == 'PRON' and token1.pos_.lower() == 'we' and token2.pos_ == 'VERB':
            claims.append((start, end))
    claims = list(set(claims))
    return claims


def find_extra_claim(token, keywords=set()):
    """
    Loosen one rule to get more non-claim examples
    """
    claims = []
    if len(token) < 5 or not any(
        [t.text.lower() in TRIGGER_WORDS.union({'we', 'our'}) for t in token]
    ):
        return claims
    n_token = len(token)

    noun_phrase_position = find_noun_phrase_offset(token)
    for (start, end) in noun_phrase_position:
        token1 = token[min(end, n_token - 1)]
        token2 = token[min(end + 1, n_token - 1)]
        tag_list = [t.text.lower() for t in token[start:end]]
        is_we_in = any([w in tag_list for w in ['we', 'our', 'paper', 'study']])
        # loosen the rule here!
        if ((token1.lemma_ in keywords) or (token2.lemma_ in keywords)) and is_we_in:
            claims.append((start, end))
        if token1.pos_ == 'ADV' and token2.lemma_ in keywords:
            claims.append((start, end + 1))
        # example: In this paper, we discuss a web-first approach
        if token1.pos_ == 'PUNCT' and (token2.pos_ == 'PRON' and token2.text.lower() == 'we'):
            claims.append((start, end + 1))
        # example: In this manuscript we produce and analyze ...
        if token1.pos_ == 'PRON' and token1.text.lower() == 'we' and token2.pos_ == 'VERB':
            claims.append((start, end))
    claims = list(set(claims))
    return claims