# GPT tutorial 1 in Python

This folder contains the guided notebooks. Start with [Lesson 1: read and inspect the training text](lesson_01_dataset.ipynb), continue to [Lesson 2: character-level tokenization](lesson_02_character_tokenization.ipynb), [Lesson 3: prepare next-token training examples](lesson_03_training_examples.ipynb), [Lesson 4: a first language model](lesson_04_bigram_model.ipynb), and [Lesson 5: train the bigram model](lesson_05_train_bigram.ipynb). The original tutorial notebook remains at [`../gpt_dev.ipynb`](../gpt_dev.ipynb) for reference. Each guided notebook links to the equivalent C and Java code.

## Set up the Python environment

Install Microsoft's **Python** and **Jupyter** extensions in VS Code. From this folder, create the project environment and install the notebook kernel:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name gpt1-python --display-name "GPT tutorial 1 (.venv, Python 3.13)"
```

Open the repository root or `gpt-1/` in VS Code. In each notebook, select the project `.venv` as its kernel. The notebooks locate `data/input.txt` when started from the repository root, `gpt-1/`, or this Python folder. The requirements include `ipykernel` for VS Code notebooks, NumPy for PyTorch interoperability, and the CPU build of PyTorch used by Lessons 3–5. Lessons 1 and 2 do not need PyTorch; Lesson 3 introduces tensors and mini-batches, Lesson 4 uses an embedding table, cross-entropy, and sampling, and Lesson 5 adds autograd and AdamW. The 100-step training run uses a tiny model and is intended to run locally on a CPU. The PyTorch CPU wheel is pinned in `requirements.txt`; its platform options are documented on [PyTorch's official install page](https://pytorch.org/get-started/locally/).

If VS Code does not find the kernel in an existing profile, use an **Empty Profile** and install Microsoft's Python and Jupyter extensions there.

## How the lessons are divided

Lesson 1 has one code cell for loading and inspecting the corpus. Lesson 2 repeats the small data-loading setup so it runs on its own, then uses three code cells: build the vocabulary, encode/decode a toy phrase, and encode the corpus. Lesson 3 repeats setup, then uses one cell each to split the corpus, inspect context/target alignment, and sample a batch. Lesson 4 builds a bigram model, measures its predictions, and generates an untrained sample. Lesson 5 records fixed baseline losses, trains for 100 mini-batches, and samples from the trained model. The splits follow the concepts rather than the source notebook's exact cell boundaries.

C and Java use source files and classes instead of notebook cells. Their numbered steps match the Python lesson concepts and can be selected with **Terminal → Run Task → Run tutorial lesson**. See the [shared lesson map](../README.md#run-lessons-by-step) for the C and Java commands.

Lesson 2 intentionally uses Python lists so the character-to-ID mapping stays visible. Lesson 3 introduces PyTorch tensors when slicing and stacking batches make the abstraction useful. Lesson 4 uses PyTorch's embedding table, cross-entropy, and multinomial sampling; Lesson 5 adds autograd and AdamW. Their C and Java companions use ordinary arrays and explicitly implement the gradient and optimizer update.
