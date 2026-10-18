#include "bigram_model.h"
#include "dataset.h"
#include "vocabulary.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { BLOCK_SIZE = 8, BATCH_SIZE = 4 };

static int parse_part(const char *part, int *first_step, int *last_step) {
    *first_step = 1;
    *last_step = 3;
    if (strcmp(part, "1") == 0) *last_step = 1;
    else if (strcmp(part, "2") == 0) *first_step = *last_step = 2;
    else if (strcmp(part, "3") == 0) *first_step = *last_step = 3;
    else if (strcmp(part, "1-2") == 0) *last_step = 2;
    else if (strcmp(part, "1-3") != 0 && strcmp(part, "all") != 0) return 0;
    return 1;
}

static void sample_batch(const int *data, size_t data_length,
                         int *inputs, int *targets) {
    for (size_t batch = 0; batch < BATCH_SIZE; batch++) {
        size_t start = (size_t)rand() % (data_length - BLOCK_SIZE);
        for (size_t time = 0; time < BLOCK_SIZE; time++) {
            size_t index = batch * BLOCK_SIZE + time;
            inputs[index] = data[start + time];
            targets[index] = data[start + time + 1];
        }
    }
}

static void print_preview(const float *values, size_t count) {
    size_t preview_count = count < 8 ? count : 8;
    putchar('[');
    for (size_t index = 0; index < preview_count; index++) {
        printf("%s%.3f", index == 0 ? "" : ", ", values[index]);
    }
    if (preview_count < count) fputs(", ...", stdout);
    puts("]");
}

int main(int argc, char *argv[]) {
    const char *part = argc == 1 ? "all" : argv[1];
    int first_step;
    int last_step;
    if (argc > 2 || !parse_part(part, &first_step, &last_step)) {
        fprintf(stderr, "Usage: gpt1_bigram [all|1|2|3|1-2|1-3]\n");
        return EXIT_FAILURE;
    }

    const char *dataset_path = "../data/input.txt";
    size_t text_length = 0;
    char *text = dataset_read(dataset_path, &text_length);
    if (text == NULL) {
        fprintf(stderr, "Could not read %s\n", dataset_path);
        return EXIT_FAILURE;
    }

    CharacterVocabulary vocabulary;
    vocabulary_build(text, text_length, &vocabulary);
    size_t token_count = 0;
    int *token_ids = vocabulary_encode(text, text_length, &vocabulary, &token_count);
    free(text);
    if (token_ids == NULL) {
        fprintf(stderr, "Could not encode the corpus.\n");
        return EXIT_FAILURE;
    }

    size_t train_length = token_count * 9 / 10;
    int inputs[BATCH_SIZE * BLOCK_SIZE];
    int targets[BATCH_SIZE * BLOCK_SIZE];
    srand(1337);
    sample_batch(token_ids, train_length, inputs, targets);

    BigramLanguageModel model = {0};
    if (!bigram_model_init(&model, vocabulary.size, 1337)) {
        fprintf(stderr, "Could not initialize the bigram model.\n");
        free(token_ids);
        return EXIT_FAILURE;
    }

    size_t logits_count = BATCH_SIZE * BLOCK_SIZE * vocabulary.size;
    float *logits = malloc(logits_count * sizeof(*logits));
    if (logits == NULL ||
        !bigram_model_forward(&model, inputs, BATCH_SIZE, BLOCK_SIZE, logits)) {
        fprintf(stderr, "Could not calculate model logits.\n");
        free(logits);
        bigram_model_destroy(&model);
        free(token_ids);
        return EXIT_FAILURE;
    }

    if (first_step <= 1) {
        size_t first_input_id = (size_t)inputs[0];
        printf("Step 1 - Look up next-character scores\n");
        printf("Vocabulary size: %zu; parameter table: [%zu, %zu] (%zu scores)\n",
               vocabulary.size, vocabulary.size, vocabulary.size,
               vocabulary.size * vocabulary.size);
        printf("Input batch: [%d, %d]; logits: [%d, %d, %zu]\n",
               BATCH_SIZE, BLOCK_SIZE, BATCH_SIZE, BLOCK_SIZE, vocabulary.size);
        printf("First input token ID %zu selects table row; first 8 logits: ", first_input_id);
        print_preview(logits, vocabulary.size);
    }

    if (first_step <= 2 && last_step >= 2) {
        double loss = bigram_model_cross_entropy(&model, logits, targets,
                                                 BATCH_SIZE, BLOCK_SIZE);
        puts("Step 2 - Compare logits with next-character targets");
        printf("Targets: %d positions (%d batch rows x %d time steps)\n",
               BATCH_SIZE * BLOCK_SIZE, BATCH_SIZE, BLOCK_SIZE);
        printf("Mean cross-entropy loss before training: %.4f\n", loss);
        puts("For each position: logsumexp(logits) - target_logit, then average.");
    }

    if (last_step >= 3) {
        int generated_ids[101];
        bigram_model_generate(&model, 0, 100, generated_ids);
        char *generated_text = vocabulary_decode(generated_ids, 101, &vocabulary);
        if (generated_text == NULL) {
            fprintf(stderr, "Could not decode generated token IDs.\n");
            free(logits);
            bigram_model_destroy(&model);
            free(token_ids);
            return EXIT_FAILURE;
        }
        puts("Step 3 - Sample characters autoregressively (untrained model)");
        printf("Start token ID: 0; generated text:\n%s\n", generated_text);
        puts("This model uses only the immediately previous character as context.");
        free(generated_text);
    }

    free(logits);
    bigram_model_destroy(&model);
    free(token_ids);
    return EXIT_SUCCESS;
}
