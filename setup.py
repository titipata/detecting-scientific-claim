from setuptools import setup, find_packages


NAME = "DETECTING-SCIENTIFIC-CLAIM"
VERSION = "1.0.0"
DESCRIPTION = "Detecting Scientific Claims in Scientific literature."
URL = "https://github.com/titipata/detecting-scientific-claim"

EXCLUDE_LIST = [
    "annotation_tool.*", "baseline.*", "experiments.*", "pubmed-rct.*",
    "static_html.*", "arxivbot.*"
]
REQUIRED = [
    "torch", "allennlp==0.9.0", "spacy", "fastText",
    "pandas",
]

if __name__ == "__main__":
    setup(
        name=NAME,
        version=VERSION,
        packages=find_packages(exclude=EXCLUDE_LIST),
        description=DESCRIPTION,
        install_requires=REQUIRED,
        url=URL,
    )
