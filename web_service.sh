python -m allennlp.service.server_simple \
    --archive-path model.tar.gz \
    --predictor discourse_predictor \
    --static-dir static_html \
    --include-package discourse \
    --title "Discourse Prediction" \
    --field-name sentence
