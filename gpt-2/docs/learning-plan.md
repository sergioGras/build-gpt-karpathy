# GPT-2 learning plan

The source material is [Karpathy's reference checkout](../build-nanogpt/README.md) and the supplied [transcript summary](summary%20transcript.txt). Use the summary as a map and check details against the actual code and original documentation. Historical speed and cost figures are observations from the tutorial, not promises for our hardware.

## Milestones

| Phase | What we will demonstrate |
| --- | --- |
| Bridge from Tutorial 1 | [Lesson 1](../python/notebooks/lesson_01_gpt1_to_gpt2.ipynb) identifies GPT-2's differences, traces dimensions, and estimates parameter count |
| Tokenization | [Lesson 2](../python/notebooks/lesson_02_gpt2_tokenization.ipynb) compares byte-level BPE behavior in Python, C, and Java and introduces document boundaries |
| Token data pipeline | [Lesson 3](../python/notebooks/lesson_03_token_streams_and_batches.ipynb) separates train/validation streams, shifts token windows into next-token batches, and explains loader cursor movement |
| Model inputs | [Lesson 4](../python/notebooks/lesson_04_token_and_position_embeddings.ipynb) traces learned token and position table lookups and the initial residual stream |
| Causal attention | [Lesson 5](../python/notebooks/lesson_05_causal_self_attention.ipynb) builds Q/K/V, scaled scores, the future-token mask, softmax weights, and value mixing |
| Decoder block | [Lesson 6](../python/notebooks/lesson_06_decoder_block.ipynb) combines LayerNorm, causal attention, the 4C MLP with GELU, and the two residual updates |
| Complete forward pass | [Lesson 7](../python/notebooks/lesson_07_tiny_gpt_forward.ipynb) connects embeddings, a decoder block, final LayerNorm, tied output weights, and next-token logits |
| Reconstruct and verify | Progress from embeddings through attention and the decoder, then load pretrained weights and compare logits against a known implementation |
| Establish training correctness | Overfit a batch, inspect initialization and weight tying, then train on fresh batches |
| Prepare datasets | Explain byte-level BPE, document boundaries, shards, and validation splits |
| Record experiments | Run a small training experiment with metrics, samples, and checkpoints; test resume |
| Use the GPU lab | After local training checks pass, validate SSH, metrics, checkpoint retrieval, and cleanup with a short capped run |
| Train GPT-2 124M | Complete a measured from-scratch run on the tutorial's FineWeb-Edu 10B-token sample, with loss curves, samples, and evaluation |
| Experiment with data | Reuse the run workflow for smaller multilingual, numerical, and financial-sequence experiments, recording representation choices and provenance |
| Optimize and evaluate | Measure selected precision/attention/compilation changes, gradient accumulation, and evaluation; keep large distributed runs optional |

These phases can span several lessons. Each lesson links the notebook, reusable implementation, C/Java counterpart, run instructions, references, and a small exercise. Use shared inputs and numeric tolerances when comparing implementations; sampled text alone is not a numerical correctness check.

## Language goals

Python leads the complete model and pretraining workflow. C and Java aim to run actual pretrained GPT-2 inference through suitable libraries and to explore training in small, readable examples. Check library compatibility with a working prototype before committing to it. Manual primitive examples should explain a concept without turning this course into an autodiff-engine project.

For Lesson 2, Python uses OpenAI `tiktoken` 0.14.x, Java uses JTokkit 1.1.0's `r50k_base` encoding, and the C companion uses `tiktoken-c` 0.9.1. The C release is installed locally by a checksum-verified script and remains outside version control. These wrappers let us compare the same IDs while briefly examining each language's abstractions and memory-ownership rules.

Notebook cells introduce ideas; reusable modules expose the operations that scripts need. Avoid importing a final training script with top-level execution. Add functionality as its lesson arrives rather than implementing the rest of the course in advance.

## Experiments are part of the course

Maintain a known baseline, then let a new configuration describe a variation. Make datasets and model settings explicit, save a run record, and compare measured behavior. Playful datasets and architectural modifications should use the same workflow as tutorial examples. Start with readable files and functions; add factories, sweeps, or external tracking services only when actual experiments justify them.

## Compute and completion

Distinguish configuration inspection, reduced local checks, short GPU training experiments, and full pretraining reproduction. Matching the GPT-2 architecture, loading a pretrained checkpoint, and training a model from random weights are different achievements. Label results with the configuration and data actually used.

The learning path aims to complete a measured GPT-2 124M-scale run, not only explain its architecture. Follow the [GPT-2 remote-compute plan](remote-compute-plan.md) for the earlier GPU pilot, the full-run estimate, and repeatable custom-data experiments. A paid run will use a concrete spend cap and cleanup plan; the scaffold does not provision resources.
