# GPT tutorial 1 in Python

This folder contains our guided Python lessons. The original notebook is kept at [`../gpt_dev.ipynb`](../gpt_dev.ipynb) as a reference; start with [`lesson_01_dataset.ipynb`](lesson_01_dataset.ipynb), which adds explanations and links to the matching C and Java code.

## Open and run in VS Code

Open the repository root folder in VS Code, open this notebook, and select the `.venv` Python kernel. The workspace setting keeps notebook paths relative to the repository root.

Create the environment once in PowerShell from this folder:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The first lesson uses the Python standard library. `ipykernel` connects this virtual environment to the notebook editor. PyTorch will be added when the tutorial reaches tensor operations.
