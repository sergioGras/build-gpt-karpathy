# GPT tutorial 1

Start with the guided [Python lessons](python/README.md), then follow their links to the corresponding C and Java code. The original tutorial notebook remains available as [`gpt_dev.ipynb`](gpt_dev.ipynb).

- Shared Tiny Shakespeare data: [`data/input.txt`](data/input.txt)
- Python notebooks: [`python/`](python/README.md)
- C project: [`c/`](c/README.md)
- Java Maven project: [`java/`](java/README.md)

## Run lessons by step

The Python notebooks use concept-sized cells. C and Java use conventional source files and entry points, so a lesson's numbered steps group naturally into functions and classes instead of pretending each source file is a notebook cell. In all three versions, the step numbers describe the same learning progression.

### Lesson 1: Read and inspect the training text

| Step | What it does | Python | C | Java |
| --- | --- | --- | --- | --- |
| 1 | Find the shared corpus | `dataset_path` in [`lesson_01_dataset.ipynb`](python/lesson_01_dataset.ipynb) | `main.c` checks the file | `TinyShakespeareDataset.findDefaultPath` searches common launch folders |
| 2 | Read the corpus | `Path.read_text` | `dataset_read` | `TinyShakespeareDataset.load` |
| 3 | Count and preview the text | `len` and slicing | `byte_count` and `fwrite` | `characterCount` and `preview` |

### Lesson 2: Character-level tokenization

| Step | What it does | Python | C | Java |
| --- | --- | --- | --- | --- |
| 1 | Build a stable character vocabulary | `sorted(set(text))` in [`lesson_02_character_tokenization.ipynb`](python/lesson_02_character_tokenization.ipynb) | `vocabulary_build` in `vocabulary.c` | `CharacterVocabulary.fromText` |
| 2 | Encode and decode a toy string | `stoi`, `itos`, `encode`, `decode` | `vocabulary_encode` and `vocabulary_decode` | `CharacterVocabulary.encode` and `.decode` |
| 3 | Encode and round-trip the full corpus | `encoded_text` list | `tokenization_main.c` | `CharacterTokenizationLesson` |

### Lesson 3: Prepare next-token training examples

| Step | What it does | Python | C | Java |
| --- | --- | --- | --- | --- |
| 1 | Split tokens into training and validation regions | tensor slices in [`lesson_03_training_examples.ipynb`](python/lesson_03_training_examples.ipynb) | split index in `batches_main.c` | `Arrays.copyOfRange` in `NextTokenBatchLesson` |
| 2 | Shift a context window to make next-token targets | `x` and `y` tensor slices | adjacent IDs in `batches_main.c` | `window` and `contextIds` |
| 3 | Sample a mini-batch of windows | `get_batch` and `torch.stack` | row-major `int` buffers | `int[][]` and `System.arraycopy` |

### Lesson 4: A first language model — the bigram model

| Step | What it does | Python | C | Java |
| --- | --- | --- | --- | --- |
| 1 | Look up next-character logits from the current token | `nn.Embedding` in [`lesson_04_bigram_model.ipynb`](python/lesson_04_bigram_model.ipynb) | `bigram_model_forward` | `BigramLanguageModel.forward` |
| 2 | Measure next-token predictions with cross-entropy | `F.cross_entropy` | stable log-sum-exp in `bigram_model.c` | stable log-sum-exp in `BigramLanguageModel` |
| 3 | Generate a continuation by sampling one token at a time | `generate` and `torch.multinomial` | `bigram_model_generate` | `BigramLanguageModel.generate` |

### Lesson 5: Train the bigram model

| Step | What it does | Python | C | Java |
| --- | --- | --- | --- | --- |
| 1 | Measure initial loss on fixed train/validation batches | `evaluate` in [`lesson_05_train_bigram.ipynb`](python/lesson_05_train_bigram.ipynb) | fixed batches in `bigram_training_main.c` | fixed batches in `BigramTrainingLesson` |
| 2 | Train for 100 mini-batches and compare loss | PyTorch autograd and `torch.optim.AdamW` | explicit gradient and AdamW in `bigram_model.c` | explicit gradient and AdamW in `BigramLanguageModel.trainBatch` |
| 3 | Generate text from the trained model | `generate` and `torch.multinomial` | `bigram_model_generate` | `BigramLanguageModel.generate` |

### Lesson 6: The mathematical trick in self-attention

| Step | What it does | Python | C | Java |
| --- | --- | --- | --- | --- |
| 1 | Average each vector with its visible past using loops | loop in [`lesson_06_self_attention_math.ipynb`](python/lesson_06_self_attention_math.ipynb) | `average_visible_past` | `averageVisiblePast` |
| 2 | Express the causal average as lower-triangular matrix multiplication | `torch.tril` and `@` | `make_causal_average_weights` and `apply_weights` | `causalAverageWeights` and `applyWeights` |
| 3 | Mask future positions and normalize the remaining scores with softmax | `masked_fill` and `F.softmax` | `softmax_rows` | `softmaxRows` |
| 4 | Use query-key similarity to make input-dependent attention weights | query/key projections and weighted values | query/key projections and weighted values | query/key projections and weighted values |

### Lesson 7: Why attention scores are scaled

| Step | What it does | Python | C | Java |
| --- | --- | --- | --- | --- |
| 1 | Measure query-key score variance at increasing head sizes | exact sign-vector enumeration in [`lesson_07_attention_scaling.ipynb`](python/lesson_07_attention_scaling.ipynb) | sign-vector enumeration | sign-vector enumeration |
| 2 | Compare softmax at ordinary and exaggerated score scales | `F.softmax` and entropy | stable softmax and entropy | stable softmax and entropy |
| 3 | Divide concrete query-key scores by the square root of head size | scaled logits and weights | scaled logits and weights | scaled logits and weights |

### Lesson 8: Layer normalization

| Step | What it does | Python | C | Java |
| --- | --- | --- | --- | --- |
| 1 | Inspect per-token statistics across channels | `mean` and `var` in [`lesson_08_layer_norm.ipynb`](python/lesson_08_layer_norm.ipynb) | `row_statistics` in `layer_norm_main.c` | `rowStatistics` in `LayerNormLesson` |
| 2 | Center and rescale each token vector | explicit formula using PyTorch tensors | loops over channel vectors | loops over `double[]` channel vectors |
| 3 | Apply a scale and shift per channel | manual formula checked against `nn.LayerNorm` | `gamma * x_hat + beta` | `gamma * x_hat + beta` |

### Lesson 9: Build one causal self-attention head

| Step | What it does | Python | C | Java |
| --- | --- | --- | --- | --- |
| 1 | Project token vectors into queries, keys, and values | three `nn.Linear` layers in [lesson_09_causal_attention_head.ipynb](python/lesson_09_causal_attention_head.ipynb) | fixed matrix projections in `attention_head_main.c` | fixed matrix projections in `CausalAttentionHeadLesson` |
| 2 | Compare every query with every key and scale the scores | `Q @ K.transpose(-2, -1) / sqrt(head_size)` | explicit dot-product loops | explicit dot-product loops |
| 3 | Mask future positions and normalize scores with softmax | triangular mask and `F.softmax` | masked scores and stable softmax loops | masked scores and stable softmax loops |
| 4 | Mix value vectors and check that future tokens cannot affect the past | weighted sum and perturbation check | weighted sum and perturbation check | weighted sum and perturbation check |

### Lesson 10: Multi-head self-attention

| Step | What it does | Python | C | Java |
| --- | --- | --- | --- | --- |
| 1 | Run two independent causal attention heads | `ModuleList` of `CausalAttentionHead` modules in [lesson_10_multi_head_attention.ipynb](python/lesson_10_multi_head_attention.ipynb) | independent query/key/value weight sets in `multi_head_attention_main.c` | independent query/key/value weight sets in `MultiHeadAttentionLesson` |
| 2 | Concatenate each head's output channels | `torch.cat` along the last dimension | copy head channels into a wider vector | copy head channels into a wider array |
| 3 | Project the combined channels back to model width | learned `Linear` plus `Dropout` | fixed projection matrix in evaluation mode | fixed projection matrix in evaluation mode |
| 4 | Check causality and compare training/evaluation dropout behavior | perturb a future token; toggle train/eval | perturb a future token | perturb a future token |

### Lesson 11: Assemble a pre-LayerNorm Transformer block

| Step | What it does | Python | C | Java |
| --- | --- | --- | --- | --- |
| 1 | Normalize each token before the attention sublayer | `nn.LayerNorm` in [lesson_11_transformer_block.ipynb](python/lesson_11_transformer_block.ipynb) | `layer_norm` | `layerNorm` |
| 2 | Add the attention output to the original input | tensor residual addition | array residual addition | array residual addition |
| 3 | Normalize again, then expand, apply ReLU, and project back | `nn.Sequential` feed-forward network | `feed_forward` | `FeedForward.forward` |
| 4 | Add the feed-forward residual and verify position-wise computation | tensor residual and perturbation check | array residual and perturbation check | array residual and perturbation check |

The attention branch uses the fixed Lesson 10 output for the shared input so this lesson can focus on pre-LayerNorm ordering, residual paths, and the position-wise feed-forward network.
### Lesson 12: Add token and position embeddings

| Step | What it does | Python | C | Java |
| --- | --- | --- | --- | --- |
| 1 | Treat token IDs as vocabulary lookup addresses | `token_ids` in [lesson_12_token_position_embeddings.ipynb](python/lesson_12_token_position_embeddings.ipynb) | `TOKEN_IDS` | `TOKEN_IDS` |
| 2 | Look up a vector for each token ID | `nn.Embedding` | `token_table[token_id]` | `tokenTable[tokenId]` |
| 3 | Look up a vector for each sequence position | position embedding lookup | `position_table[position]` | `positionTable[position]` |
| 4 | Add token and position vectors before the Transformer blocks | broadcasted tensor addition | channel-wise array addition | channel-wise array addition |

### Lesson 13: Assemble a tiny decoder language model

| Step | What it does | Python | C | Java |
| --- | --- | --- | --- | --- |
| 1 | Add token and position embeddings to form the block input | `nn.Embedding` in [lesson_13_decoder_language_model.ipynb](python/lesson_13_decoder_language_model.ipynb) | embedding tables in `decoder_model_main.c` | embedding tables in `DecoderLanguageModelLesson` |
| 2 | Pass representations through two pre-LayerNorm blocks | `TransformerBlock` modules | forward-pass functions in `decoder_model_main.c` | `forward` and array helpers in `DecoderLanguageModelLesson` |
| 3 | Project to vocabulary logits and calculate next-token cross-entropy | `nn.Linear` and `F.cross_entropy` | vocabulary projection and stable log-sum-exp | vocabulary projection and stable log-sum-exp |
| 4 | Generate one token at a time using the last-position logits | cropped context and fixed probability thresholds | cropped context and fixed probability thresholds | cropped context and fixed probability thresholds |

### Lesson 14: Train the decoder language model

| Step | What it does | Python | C | Java |
| --- | --- | --- | --- | --- |
| 1 | Read Tiny Shakespeare, split train/validation text, and sample windows | data and `get_batch` in [lesson_14_train_decoder_language_model.ipynb](python/lesson_14_train_decoder_language_model.ipynb) | corpus/tokenization mechanics in Lessons 1–3 | corpus/tokenization mechanics in Lessons 1–3 |
| 2 | Train all decoder parameters with backpropagation and AdamW | `loss.backward()` and `optimizer.step()` | explicit gradient and AdamW mechanics in [Lesson 5](c/src/bigram_training_main.c) | explicit gradient and AdamW mechanics in [Lesson 5](java/src/main/java/learning/karpathy/gpt1/BigramTrainingLesson.java) |
| 3 | Plot train/validation loss and sample from the updated model | Matplotlib curve and autoregressive generation | Lesson 13 shows decoder generation with fixed weights | Lesson 13 shows decoder generation with fixed weights |

This first full-Transformer training loop is implemented in Python with PyTorch autograd. C and Java show the matching decoder forward pass in Lesson 13 and implement the cross-entropy gradient and AdamW update explicitly for the bigram model in Lesson 5; keeping those smaller examples avoids duplicating a full autodiff engine in this tutorial.


In VS Code, open this repository or the gpt-1 folder and choose **Terminal → Run Task → Run tutorial lesson**. Select lesson 1 through 13, then c or java, followed by the steps to show. Lessons 1–5 and 7–8 have three steps; Lessons 6 and 9–13 have four, so those menus also accept 4, 1-4, 2-4, and 3-4. Lesson 14 is the Python notebook; open it and use **Run All**. A selected C/Java step runs any calculations it needs while hiding earlier output.

The same workflow is available from a PowerShell terminal at the repository root:

```powershell
.\scripts\run-lesson.ps1 -Lesson 2 -Language c -Part 1-2
.\scripts\run-lesson.ps1 -Lesson 2 -Language java -Part 3
.\scripts\run-lesson.ps1 -Lesson 3 -Language java -Part all
.\scripts\run-lesson.ps1 -Lesson 4 -Language c -Part 2
.\scripts\run-lesson.ps1 -Lesson 5 -Language java -Part all
.\scripts\run-lesson.ps1 -Lesson 6 -Language c -Part 4
.\scripts\run-lesson.ps1 -Lesson 7 -Language java -Part all
.\scripts\run-lesson.ps1 -Lesson 8 -Language c -Part all
.\scripts\run-lesson.ps1 -Lesson 8 -Language java -Part all
.\scripts\run-lesson.ps1 -Lesson 9 -Language c -Part 3-4
.\scripts\run-lesson.ps1 -Lesson 9 -Language java -Part all
.\scripts\run-lesson.ps1 -Lesson 10 -Language c -Part all
.\scripts\run-lesson.ps1 -Lesson 10 -Language java -Part all
.\scripts\run-lesson.ps1 -Lesson 11 -Language c -Part all
.\scripts\run-lesson.ps1 -Lesson 11 -Language java -Part all
.\scripts\run-lesson.ps1 -Lesson 12 -Language c -Part all
.\scripts\run-lesson.ps1 -Lesson 12 -Language java -Part all
.\scripts\run-lesson.ps1 -Lesson 13 -Language c -Part all
.\scripts\run-lesson.ps1 -Lesson 13 -Language java -Part 4
```

For the full decoder training loop, open [`lesson_14_train_decoder_language_model.ipynb`](python/lesson_14_train_decoder_language_model.ipynb) and run its cells in order.

The script uses CMake and the Maven Wrapper. It finds this machine's MSYS2 CMake and JDK 25 installations, or uses tools already on `PATH`/`JAVA_HOME`. If Java 25 is installed elsewhere, pass `-JavaHome "C:\path\to\jdk-25"`.

Python iterates Unicode code points, Java's `char` represents UTF-16 code units, and this C lesson maps bytes. The shared corpus currently contains ASCII only, so all three versions assign the same vocabulary order and token IDs. Unicode-aware C tokenization is a later extension if the lessons need it.

Lesson 3 adds the CPU build of PyTorch for tensor slicing, random index selection, and stacking windows into batches. Lesson 4 uses PyTorch's embedding, cross-entropy, and sampling operations; Lesson 5 adds autograd and AdamW for the bigram model. Lessons 6 and 7 study self-attention and score scaling; Lesson 8 covers per-token LayerNorm. Lesson 9 builds one causal attention head. Lesson 10 runs independent heads, concatenates their channels, and projects back to model width. Lesson 11 combines pre-LayerNorm, residual connections, and a position-wise feed-forward network into one Transformer block. Lesson 12 adds token and position embeddings. Lesson 13 joins these pieces into a tiny decoder language model, computes logits and loss, and generates a short continuation with fixed untrained weights. Lesson 14 trains a smaller full decoder with PyTorch autograd and plots train/validation loss. This configuration runs locally on CPU; a later larger run will tell us when a rented GPU is useful.
