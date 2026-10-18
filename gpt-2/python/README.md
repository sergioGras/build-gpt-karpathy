# Python companion

The notebooks are the learning entry point. Reusable code lives in `src/gpt2_lab`, so notebooks and command-line experiments use the same implementation. The editable installation makes source changes available without reinstalling.

Start with [00_project_tour.ipynb](notebooks/00_project_tour.ipynb), then continue with [Lesson 1: From GPT-1 to GPT-2](notebooks/lesson_01_gpt1_to_gpt2.ipynb), [Lesson 2: GPT-2 tokenization](notebooks/lesson_02_gpt2_tokenization.ipynb), [Lesson 3: Token streams and batches](notebooks/lesson_03_token_streams_and_batches.ipynb), [Lesson 4: Token and position embeddings](notebooks/lesson_04_token_and_position_embeddings.ipynb), [Lesson 5: Causal self-attention](notebooks/lesson_05_causal_self_attention.ipynb), [Lesson 6: The GPT-2 decoder block](notebooks/lesson_06_decoder_block.ipynb), and [Lesson 7: A complete tiny GPT forward pass](notebooks/lesson_07_tiny_gpt_forward.ipynb). Lesson 7 connects token and position lookup to one decoder block, final normalization, and tied vocabulary logits; it links each stage to readable C and Java companions.

## Setup on Windows

From the repository root:

```powershell
.\scripts\setup-gpt2-python.ps1
.\scripts\run-gpt2.ps1 -Language python
```

In VS Code choose **Karpathy GPT-2 (project .venv)** as the notebook kernel, or select `gpt-2/python/.venv/Scripts/python.exe` from Python Environments. This environment is separate from GPT-1. The setup script installs this package, notebook support, the tokenizer used in Lesson 2, NumPy, and the CPU build of PyTorch used in Lessons 5 and 6.

## Direct commands and Linux

From `gpt-2/python`:

```bash
python -m venv .venv
# Linux/macOS interpreter: .venv/bin/python
# Windows interpreter: .venv/Scripts/python.exe
.venv/bin/python -m pip install -e '.[notebooks]'
.venv/bin/python -m pip install 'numpy>=2.1,<3'
# CPU-only lesson environment; use PyTorch's selector for a GPU build.
.venv/bin/python -m pip install --index-url https://download.pytorch.org/whl/cpu -e '.[ml]'
.venv/bin/python -m ipykernel install --user --name karpathy-gpt2 --display-name 'Karpathy GPT-2 (project .venv)'
.venv/bin/python -m gpt2_lab plan --config ../experiments/configs/tiny-playground.json
.venv/bin/python -m unittest discover -s tests -v
```

On Windows, the setup helper installs the CPU PyTorch wheel from the official PyTorch package index. For a GPU training environment, choose the matching CUDA build using [PyTorch's current install selector](https://pytorch.org/get-started/locally/) when the cloud lessons reach that point. Both `python -m gpt2_lab` and the installed `gpt2-lab` command expose `plan` and `prepare-run`. `plan` reads settings without creating a run. `prepare-run` saves settings and provenance; neither command trains a model. See the [experiment guide](../experiments/README.md).

## Growing the implementation

Keep imports free of training, dataset downloads, and cloud actions. Add model, data, training, and evaluation modules as their lessons introduce them. Keep the mathematical example visible in the notebook, then link to the reusable implementation. An architectural experiment can add a named implementation once it needs one; we do not need a plugin system before that.
