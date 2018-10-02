# Sentence Annotation Tool (WIP)

Along with discourse and claim prediction tool, we also release
annotation tool that we made to collect claim dataset.


### Annotation Tool Web Service

All parameters are located in `params.yaml` include the following

- `pmids_path` is a path to list of PMIDs or JSON file of PMIDs that you want to tag, 
default as `data/pmids.json` or `data/pmids.txt`. JSON file has to have the following keys: 
`paper_id`, `title`, `abstract`, `sentences`
- `labels` is a list of labels that you want to annotate
- `output_path` is a path to output file, default as `data/labels.json`
- `store_details` is an integer defined if we want to store full tags information or only labels

The instructions for annotators can be edited in `flask_templates/index.html`.
After editing YAML file, you can run `flask` to start the annotation tool.

```bash
export FLASK_APP=main.py
flask run --host=0.0.0.0 --port=5555 # this will serve at port 5555
```

The data will be saved in default location at `data/labels.json` where
each lines contains JSON of title, abstract, sentences, and labels.
