# Sentence Annotation Tool (WIP)

Along with discourse and claim prediction tool, we also release
annotation tool that we made to collect claim dataset.


### Annotation Tool Web Service

First, list PMIDs that you want to tag in `data/pmids.txt` and label that you
want to annotate `data/labels.txt`. The instructions can be edited in `flask_templates/index.html`

After edit 3 files, you can run flask to start the annotation tool.

```bash
export FLASK_APP=main.py
flask run --host=0.0.0.0 # this will serve at port 5000
```
