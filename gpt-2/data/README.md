# Datasets

`samples/playground.txt` is a small fictional corpus for checking file paths and experiment setup. It is not a pretraining dataset or a tokenizer implementation.

Put custom local text under `local/`, and future downloads/token shards under `cache/`. These two directories are ignored by Git. Configurations name the dataset explicitly; there are no automatic large downloads during setup, imports, or configuration inspection.

We will add tokenizer choice, document boundaries, splitting, and sharding as the data lessons introduce them. Custom datasets should go through the same data interface as the tutorial corpus.
