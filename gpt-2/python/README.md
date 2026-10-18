# Python companion

The notebooks are the learning entry point. Reusable code lives in `src/gpt2_lab`, so notebooks and command-line experiments use the same implementation. The editable installation makes source changes available without reinstalling.

Start with [00_project_tour.ipynb](notebooks/00_project_tour.ipynb), then continue with [Lesson 1: From GPT-1 to GPT-2](notebooks/lesson_01_gpt1_to_gpt2.ipynb), [Lesson 2: GPT-2 tokenization](notebooks/lesson_02_gpt2_tokenization.ipynb), and [Lesson 3: Token streams and batches](notebooks/lesson_03_token_streams_and_batches.ipynb). Lesson 2 uses `tiktoken` for the GPT-2 vocabulary; Lesson 3 splits the toy documents and forms shifted next-token batches.

## Setup on Windows

From the repository root:

```powershell
.\scripts\setup-gpt2-python.ps1
.\scripts\run-gpt2.ps1 -Language python
```

In VS Code choose **Karpathy GPT-2 (project .venv)** as the notebook kernel, or select `gpt-2/python/.venv/Scripts/python.exe` from Python Environments. This environment is separate from GPT-1. The setup script installs this package, notebook support, and the tokenizer used in Lesson 2. Model and training libraries will be added when a lesson needs them.

## Direct commands and Linux

From `gpt-2/python`:

```bash
python -m venv .venv
# Linux/macOS interpreter: .venv/bin/python
# Windows interpreter: .venv/Scripts/python.exe
.venv/bin/python -m pip install -e '.[notebooks]'
.venv/bin/python -m ipykernel install --user --name karpathy-gpt2 --display-name 'Karpathy GPT-2 (project .venv)'
.venv/bin/python -m gpt2_lab plan --config ../experiments/configs/tiny-playground.json
.venv/bin/python -m unittest discover -s tests -v
```

Both `python -m gpt2_lab` and the installed `gpt2-lab` command expose `plan` and `prepare-run`. `plan` reads settings without creating a run. `prepare-run` saves settings and provenance; neither command trains a model. See the [experiment guide](../experiments/README.md).

## Growing the implementation

Keep imports free of training, dataset downloads, and cloud actions. Add model, data, training, and evaluation modules as their lessons introduce them. Keep the mathematical example visible in the notebook, then link to the reusable implementation. An architectural experiment can add a named implementation once it needs one; we do not need a plugin system before that.
