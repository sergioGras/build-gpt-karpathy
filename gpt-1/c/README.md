# GPT tutorial 1 in C

This is the C companion project for the guided [Python lessons](../python/README.md). It uses standard C17 and a small CMake project with conventional `src/` and `include/` folders. The [lesson map](../README.md#run-lessons-by-step) connects each numbered step to its Python and Java counterpart.

## Build and run

From this directory, configure and build with CMake, then run the dataset lesson:

```powershell
cmake --preset default
cmake --build --preset default --target gpt1_c
build/gpt1_c.exe
```

The preset selects Ninja and puts generated files in the ignored `build/` directory.

## Run a lesson step

From the repository root, use the [shared runner](../README.md#run-lessons-by-step):

```powershell
.\scripts\run-lesson.ps1 -Lesson 2 -Language c -Part 1-2
.\scripts\run-lesson.ps1 -Lesson 4 -Language c -Part all
.\scripts\run-lesson.ps1 -Lesson 5 -Language c -Part all
.\scripts\run-lesson.ps1 -Lesson 6 -Language c -Part 4
.\scripts\run-lesson.ps1 -Lesson 7 -Language c -Part all
.\scripts\run-lesson.ps1 -Lesson 8 -Language c -Part all
.\scripts\run-lesson.ps1 -Lesson 9 -Language c -Part 3-4
.\scripts\run-lesson.ps1 -Lesson 10 -Language c -Part all
.\scripts\run-lesson.ps1 -Lesson 11 -Language c -Part all
.\scripts\run-lesson.ps1 -Lesson 12 -Language c -Part all
.\scripts\run-lesson.ps1 -Lesson 13 -Language c -Part all
```

Or use **Terminal → Run Task → Run tutorial lesson** in VS Code. Choose lesson 1 for corpus reading, 2 for character tokenization, 3 for next-token batches, 4 for the untrained bigram model, 5 for bigram training, 6 for attention math, 7 for attention-score scaling, 8 for LayerNorm, 9 for a causal attention head, 10 for multi-head attention, 11 for a Transformer block, 12 for token and position embeddings, or 13 for a tiny decoder language model. Lessons 1–5 and 7–8 have three steps; Lessons 6 and 9–13 have four and accept the extended ranges 1-4, 2-4, and 3-4. A selected step calculates prerequisites while hiding earlier output.

Lesson 1 starts at `src/main.c` and delegates file reading to `dataset_read` in `src/dataset.c`. Lesson 2 starts at `src/tokenization_main.c`; `src/vocabulary.c` builds a byte-to-ID lookup table, encodes text into integer IDs, and decodes IDs back into text. Lesson 3 starts at `src/batches_main.c` and stores batches in flat arrays. Lesson 4 starts at `src/bigram_main.c`; `src/bigram_model.c` implements table lookup, stable cross-entropy, and sampling. Lesson 5 starts at `src/bigram_training_main.c` and calls `bigram_model_train_batch` to calculate the softmax-cross-entropy gradient and perform an AdamW update. Lesson 6 starts at `src/attention_math_main.c`; its arrays expose causal averaging, matrix multiplication, masked softmax, and query-key weighted values. Lesson 7 starts at `src/attention_scaling_main.c` and compares the variance and softmax behavior of unscaled and scaled dot products. Lesson 8 starts at `src/layer_norm_main.c` and calculates per-token mean/variance, normalized values, and the affine scale/shift. See the [Python Lesson 8 notebook](../python/lesson_08_layer_norm.ipynb) and [Java project](../java/README.md) for the parallel implementations.


Lesson 9 starts at `src/attention_head_main.c`. Its fixed projection matrices, score arrays, causal mask, softmax, and explicit weighted-value loops mirror the Python and Java forward pass; a final perturbation check demonstrates that future tokens cannot affect earlier outputs. C does not train these weights in this lesson.

Lesson 10 starts at `src/multi_head_attention_main.c`. It runs two independent causal heads, copies their output channels into one vector, and applies a fixed output projection. The C path models evaluation mode, where dropout is inactive, and checks that the combined output remains causal.


Lesson 11 starts at src/transformer_block_main.c. It applies LayerNorm before each sublayer, adds attention and feed-forward results through residual connections, and checks that the feed-forward network mixes channels within a token without mixing time positions. Its attention result is the fixed Lesson 10 output for the shared input; this keeps attention implementation details from obscuring the Transformer block structure.


Lesson 12 starts at `src/embedding_main.c`. It uses table rows for token identity and position, adds the vectors at each batch/time location, and checks that repeated tokens at different positions receive different model inputs.

Lesson 13 starts at `src/decoder_model_main.c`. The readable forward pass joins the embedding tables, two pre-LayerNorm blocks, a vocabulary projection, stable next-token cross-entropy, and deterministic autoregressive generation. Fixed toy weights and sampling thresholds keep the results comparable with Python and Java; this lesson does not train the model.

Lesson 14 trains the full decoder in the Python notebook with PyTorch autograd. For the C-side training mechanics, revisit Lesson 5: `bigram_training_main.c` and `bigram_model.c` derive the softmax-cross-entropy gradient and apply AdamW explicitly. Lesson 13 remains the C forward-pass counterpart for the full decoder; this project does not yet implement its complete backward pass in C.
