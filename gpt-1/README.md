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

In VS Code, open this repository or the `gpt-1` folder and choose **Terminal → Run Task → Run tutorial lesson**. Select lesson `1` through `8`, then `c` or `java`, followed by the steps to show. Lessons 1–5 and 7–8 have three steps; Lesson 6 has four, so its menu also accepts `4`, `1-4`, `2-4`, and `3-4`. A selected step runs any calculations it needs while hiding earlier output.

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
```

The script uses CMake and the Maven Wrapper. It finds this machine's MSYS2 CMake and JDK 25 installations, or uses tools already on `PATH`/`JAVA_HOME`. If Java 25 is installed elsewhere, pass `-JavaHome "C:\path\to\jdk-25"`.

Python iterates Unicode code points, Java's `char` represents UTF-16 code units, and this C lesson maps bytes. The shared corpus currently contains ASCII only, so all three versions assign the same vocabulary order and token IDs. Unicode-aware C tokenization is a later extension if the lessons need it.

Lesson 3 adds the CPU build of PyTorch for tensor slicing, random index selection, and stacking windows into batches. Lesson 4 uses PyTorch's embedding, cross-entropy, and sampling operations; Lesson 5 adds autograd and AdamW. Lessons 6 and 7 study self-attention and score scaling; Lesson 8 covers per-token LayerNorm with tiny CPU examples. The full attention/transformer training stage will determine when a rented GPU lab is useful.
