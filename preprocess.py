import os
import json
import spacy
from tqdm import tqdm

nlp = spacy.load('en_core_web_sm')


def read_pubmed_rct(file_name, desc='reading...'):
    """
    Read Pubmed 200k RCT file and tokenize using ``spacy``
    """
    tokenized_list = []
    with open(file_name, 'r') as f:
        for line in tqdm(f.readlines(), desc=desc):
            if not line.startswith('#') and line.strip() != '':
                label, sent = line.split('\t')
                tokens = nlp(sent.strip())
                text_tokens = [token.text for token in tokens]
                pos_tokens = [token.pos_ for token in tokens]
                d = {
                    'label': label,
                    'sentence': text_tokens,
                    'pos': pos_tokens,
                    'sentence_text': sent
                }
                tokenized_list.append(d)
    return tokenized_list


def save_json_list(tokenized_list, path):
    """
    Save list of dictionary to JSON
    """
    with open(path, 'w') as fp:
        fp.write('\n'.join(json.dumps(i) for i in tokenized_list))


if __name__ == '__main__':
    training_list = read_pubmed_rct(os.path.join('pubmed-rct', 'PubMed_200k_RCT', 'train.txt'))
    dev_list = read_pubmed_rct(os.path.join('pubmed-rct', 'PubMed_200k_RCT', 'dev.txt'))
    testing_list = read_pubmed_rct(os.path.join('pubmed-rct', 'PubMed_200k_RCT', 'test.txt'))

    save_json_list(training_list, os.path.join('pubmed-rct', 'PubMed_200k_RCT', 'train.json'))
    save_json_list(dev_list, os.path.join('pubmed-rct', 'PubMed_200k_RCT', 'dev.json'))
    save_json_list(testing_list, os.path.join('pubmed-rct', 'PubMed_200k_RCT', 'test.json'))
