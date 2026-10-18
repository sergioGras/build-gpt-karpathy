# Experiments and playful variations

An experiment starts with a question and a JSON configuration. The same structure is for tutorial baselines, fictional datasets, unusual architectures, and later GPU runs. Keep a baseline configuration so a change can be compared against something known.

The scaffold supports local UTF-8 text files. It validates settings, inspects a plan, and prepares a run record. Tokenization, training, evaluation, and cloud provisioning are future lessons; a successful plan does not prove a model can train on those settings.

## Try your own dataset

From the repository root:

```powershell
Copy-Item .\gpt-2\experiments\configs\tiny-playground.json .\gpt-2\experiments\configs\space-pirates.json
New-Item -ItemType Directory -Force .\gpt-2\data\local
```

Place your text in `gpt-2/data/local/space-pirates.txt`. Edit the new JSON: set `name` to `space-pirates`, write the experiment's question in `description`, and set `dataset.path` to `../../data/local/space-pirates.txt`. Change model/training settings as needed, then inspect it:

```powershell
.\scripts\run-gpt2.ps1 -Language python -Config .\gpt-2\experiments\configs\space-pirates.json
.\scripts\run-gpt2.ps1 -Language python -Action prepare-run -Config .\gpt-2\experiments\configs\space-pirates.json
```

Dataset paths are relative to the JSON file, regardless of the folder from which you launch a command. Explicit `-Config` paths are relative to your launch folder. `n_embd` must be divisible by `n_head`; misspelled settings fail validation instead of being silently ignored. The vocabulary size is currently a dimension setting only: the scaffold has no tokenizer yet.

## What a prepared run contains

Each `prepare-run` creates a fresh `gpt-2/runs/<name>-<UTC timestamp>-<unique suffix>/` with:

- `config.json`: a settings snapshot with the dataset path resolved to the actual local file.
- `manifest.json`: prepared status, configuration and dataset hashes, source Git commit/dirty flag, and Python runtime information.

No weights, metrics, or training samples are created. Later trainers will add metrics, checkpoints, and generated samples here. Prepared records are not completed experiments. A commit plus a dirty flag does not preserve uncommitted code; source snapshots and complete training/resume state will be addressed before remote training. The resolved local dataset path must be mapped to the remote upload location when a cloud runner is introduced.

`runs/`, `data/local/`, and `data/cache/` are ignored by Git. Share selected results and explanations under `docs/`; decide explicitly which datasets to commit. The tiny sample in `data/samples/` is public fictional text included for setup checks.

## Architecture changes

Start with a concrete question, such as changing context length, comparing activations, or replacing positional embeddings. Shape/training changes can use a new configuration. A new operation also needs its implementation and a focused correctness check; adding a JSON field alone cannot change model behavior. We will introduce such switches alongside the relevant lesson rather than adding unused abstractions now.

GPU launch, dataset acquisition, checkpoint resume, and sweep automation will build on this configuration/run boundary. None of these actions is triggered by inspection or preparation.
