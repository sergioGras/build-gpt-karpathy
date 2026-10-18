# GPT tutorial 1 in Python

This folder contains the guided notebooks. Start with [Lesson 1: read and inspect the training text](lesson_01_dataset.ipynb), then continue to [Lesson 2: character-level tokenization](lesson_02_character_tokenization.ipynb). The original tutorial notebook remains at [`../gpt_dev.ipynb`](../gpt_dev.ipynb) for reference. Each guided notebook links to the equivalent C and Java code.

## Set up the Python environment

Install Microsoft's **Python** and **Jupyter** extensions in VS Code. From this folder, create the project environment and install the notebook kernel:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name gpt1-python --display-name "GPT tutorial 1 (.venv, Python 3.13)"
```

Open the repository root or `gpt-1/` in VS Code. In either notebook, select the project `.venv` as its kernel. The notebooks locate `data/input.txt` when started from the repository root, `gpt-1/`, or this Python folder. The only required package is `ipykernel`, used by VS Code to run notebook cells.

If VS Code does not find the kernel in an existing profile, use an **Empty Profile** and install Microsoft's Python and Jupyter extensions there.

## How the lessons are divided

Lesson 1 has one code cell for loading and inspecting the corpus. Lesson 2 repeats the small data-loading setup so it runs on its own, then uses three code cells: build the vocabulary, encode/decode a toy phrase, and encode the corpus. The split follows the concepts rather than the source notebook's exact cell boundaries.

C and Java use source files and classes instead of notebook cells. Their numbered steps match the Python lesson concepts and can be selected with **Terminal → Run Task → Run tutorial lesson**. See the [shared lesson map](../README.md#run-lessons-by-step) for the C and Java commands.

PyTorch will be introduced when tensors and batches become useful. Lesson 2 intentionally uses Python lists so the character-to-ID mapping stays visible.
