# GPT-2 training and experiment lab

This plan makes a from-scratch GPT-2-scale training run a project goal, with custom-data experiments as a follow-on. It reuses the provider workflow from the [project remote-compute plan](../../docs/remote-compute-plan.md) and its RunPod playbook, while keeping GPT-2's datasets, lessons, and run records separate.

## Training outcomes

The capstone target is a GPT-2 124M-shaped decoder trained from random initialization on the tutorial's roughly 10-billion-token FineWeb-Edu sample, then evaluated and sampled. The upstream repository describes this optimized reproduction as about one hour and about $10; that is a historical planning estimate for its specific code, GPU setup, and run, not a guaranteed price or elapsed time for another provider or implementation.

We will also support smaller, repeatable experiments on alternative datasets. Each run should say whether it trains from scratch or continues from a checkpoint; those answer different questions and have different compute needs. Candidate experiments include multilingual text, numerical sequences, and financial time series. The data format and tokenizer are part of each experiment: GPT-2's English-oriented BPE may not be the best representation for numbers or every language.

## When cloud becomes useful

The embedding and attention lessons stay local. We will first assemble a reduced GPT model, check its forward pass and loss, and verify that a small model can overfit a batch. Those checks make mistakes cheap to diagnose.

Once the local training loop logs loss and saves a checkpoint (roughly the training-correctness lessons, likely around Lesson 10–12), run an early, short GPU Pod experiment. Its purpose is to test SSH from VS Code, launch and shutdown, metrics retrieval, and the speed difference on a small configuration. A low-cost single GPU is enough for that pilot; it does not need to train the full 10-billion-token dataset.

After the data pipeline, model, and training settings pass local checks, benchmark the intended full-run configuration on the selected hardware. Use measured tokens per second and the provider's current quote to estimate total time and cost before starting the full 124M run. The tutorial's 8×A100-scale setup is a throughput reference; it is not a requirement for our reduced lessons or custom-data tests.

## RunPod workflow

Use an interactive RunPod Pod and SSH connection for training, debugging, plotting, and checkpoint inspection. Prepare source, dataset manifests, and scripts locally before the billable session. Start without persistent storage when the required data and outputs can be transferred in and out; persistent volumes have separate charges. Keep API keys and SSH private keys on the controlling computer, outside the repository and uploads.

For every paid run, record a run ID, source revision, model and data configuration, tokenizer, expected outputs, and a hard spend and runtime ceiling. Add an expiry or watchdog and verify that the Pod is deleted after retrieval. Before the first GPU pilot, validate this path with a tiny run and confirm that logs, metrics, and a checkpoint return to the local workspace.

## Price checks and budget

Karpathy's [upstream repository README](https://github.com/karpathy/build-nanogpt) gives the approximately one-hour, $10 estimate for reproducing GPT-2 124M. RunPod's [current GPU pricing](https://www.runpod.io/pricing) is the source for a new estimate; the page currently lists A100 80GB Pods at $1.79 per GPU-hour and an A100 SXM Cluster rate of $2.15 per GPU-hour (pricing page checked 2026-10-18). Availability, cloud tier, GPU count, startup, validation, storage, and the time our own code takes can change the total. We will recalculate against live capacity and a short throughput benchmark before each full run.

The project budget can support experiments, but it is not yet a per-run limit. Set a concrete maximum amount on the run card before provisioning. The first pilot should have its own small cap; the full reproduction should have a separate cap that covers setup, training, validation, and artifact retrieval.

## Experiment loop

Keep a known baseline, then vary one factor at a time: dataset, tokenizer or representation, model size, or training settings. Save train and validation curves, sample generations, checkpoint metadata, and dataset provenance. Compare equal token budgets where that is meaningful, and label reduced runs clearly instead of presenting them as full GPT-2 reproduction.

The initial framework will stay simple: versioned configuration, a run directory, machine-readable metrics, and scripts to prepare, launch, retrieve, and delete. Add sweeps or a tracking service only when multiple real experiments make them useful.
