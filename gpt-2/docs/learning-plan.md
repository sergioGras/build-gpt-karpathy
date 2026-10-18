# GPT-2 learning plan

The source material is [Karpathy's reference checkout](../build-nanogpt/README.md) and the supplied [transcript summary](summary%20transcript.txt). Use the summary as a map and check details against the actual code and original documentation. Historical speed and cost figures are observations from the tutorial, not promises for our hardware.

## Milestones

| Phase | What we will demonstrate |
| --- | --- |
| Bridge from GPT-1 | Identify GPT-2's differences, trace dimensions, and inspect parameter organization |
| Reconstruct and verify | Build the decoder, load pretrained weights, and compare logits against a known implementation |
| Establish training correctness | Overfit a batch, inspect initialization and weight tying, then train on fresh batches |
| Prepare datasets | Explain byte-level BPE, document boundaries, shards, and validation splits |
| Record experiments | Run a small training experiment with metrics, samples, and checkpoints; test resume |
| Understand GPU execution | Measure selected precision/attention/compilation changes and gradient accumulation |
| Evaluate and scale | Explain completion scoring, run bounded evaluation, and study distributed training |

These phases can span several lessons. Each lesson links the notebook, reusable implementation, C/Java counterpart, run instructions, references, and a small exercise. Use shared inputs and numeric tolerances when comparing implementations; sampled text alone is not a numerical correctness check.

## Language goals

Python leads the complete model and pretraining workflow. C and Java aim to run actual pretrained GPT-2 inference through suitable libraries and to explore training in small, readable examples. Check library compatibility with a working prototype before committing to it. Manual primitive examples should explain a concept without turning this course into an autodiff-engine project.

Notebook cells introduce ideas; reusable modules expose the operations that scripts need. Avoid importing a final training script with top-level execution. Add functionality as its lesson arrives rather than implementing the rest of the course in advance.

## Experiments are part of the course

Maintain a known baseline, then let a new configuration describe a variation. Make datasets and model settings explicit, save a run record, and compare measured behavior. Playful datasets and architectural modifications should use the same workflow as tutorial examples. Start with readable files and functions; add factories, sweeps, or external tracking services only when actual experiments justify them.

## Compute and completion

Distinguish configuration inspection, reduced local checks, short GPU training experiments, and full pretraining reproduction. Matching the GPT-2 architecture, loading a pretrained checkpoint, and training a model from random weights are different achievements. Label results with the configuration and data actually used.

The learning path can demonstrate and explain each topic with bounded experiments. A full billions-of-tokens reproduction is an additional run with its own budget and outputs. Follow the [remote-compute plan](../../docs/remote-compute-plan.md) when that stage is reached; the scaffold does not provision resources.
