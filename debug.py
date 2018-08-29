"""
Debugging script for VSCode
See https://github.com/allenai/allennlp/blob/master/tutorials/how_to/using_a_debugger.md for the details

Basically, we can run "Debug > Start Debugging" in VSCode
"""
import json
import shutil
import sys
from allennlp.commands import main

config_file = "experiments/crf_pubmed_rct.json" # select experiments

overrides = json.dumps({"trainer": {"cuda_device": -1}})
serialization_dir = "output_crf" # saving folder
shutil.rmtree(serialization_dir, ignore_errors=True)

sys.argv = [
    "allennlp",
    "train",
    config_file,
    "-s", serialization_dir,
    "--include-package", "discourse",
    "-o", overrides,
]

main()