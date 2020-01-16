# A new expertly-annotated dataset of biomedical claims

As part of the new claim extraction task, we introduce a novel dataset of expertly annotated claims in biomedical paper abstracts. While there are multiple definitions of scientific claims proposed in previous literature, we follow the definition of previous articles to characterize a claim. This is, a claim is a statement that either

- declares something is better
- proposes something new, or
- describes a new finding or a new cause-effect relationship. One abstract can have multiple claims.

See [`annotation_tool`](https://github.com/titipata/detecting-scientific-claim/tree/master/annotation_tool) for more information on how we annotate the data. We use `spaCy` for splitting sentences from the abstract.

Three experts (with biomedical background and fluent in Enlish) are asked to annotate the data. The final dataset is done by using a majority vote between all experts.