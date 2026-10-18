#include "bigram_model.h"
#include "dataset.h"
#include "vocabulary.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const size_t BLOCK_SIZE = 8;
static const size_t BATCH_SIZE = 32;
static const size_t TRAINING_STEPS = 100;

static int parse_part(const char *part, int *first_step, int *last_step) {
    *first_step = 1;
    *last_step = 3;
    if (strcmp(part, "1") == 0) *last_step = 1;
    else if (strcmp(part, "2") == 0) *first_step = *last_step = 2;
    else if (strcmp(part, "3") == 0) *first_step = *last_step = 3;
    else if (strcmp(part, "1-2") == 0) *last_step = 2;
    else if (strcmp(part, "1-3") == 0 || strcmp(part, "all") == 0) { /* defaults */ }
    else return 0;
    return 1;
}

static void sample_batch(const int *data, size_t data_length, size_t batch_size,
                         int *inputs, int *targets) {
    for (size_t batch = 0; batch < batch_size; batch++) {
        size_t start = (size_t)rand() % (data_length - BLOCK_SIZE);
        for (size_t time = 0; time < BLOCK_SIZE; time++) {
            size_t index = batch * BLOCK_SIZE + time;
            inputs[index] = data[start + time];
            targets[index] = data[start + time + 1];
        }
    }
}

static double evaluate(const BigramLanguageModel *model, const int *inputs,
                       const int *targets, float *logits) {
    if (!bigram_model_forward(model, inputs, BATCH_SIZE, BLOCK_SIZE, logits)) {
        return -1.0;
    }
    return bigram_model_cross_entropy(model, logits, targets, BATCH_SIZE, BLOCK_SIZE);
}

static void print_loss_line(size_t step, double train_loss, double validation_loss) {
    printf("%4zu | %10.4f | %15.4f\n", step, train_loss, validation_loss);
}

int main(int argc, char *argv[]) {
    const char *part = argc == 1 ? "all" : argv[1];
    int first_step;
    int last_step;
    if (argc > 2 || !parse_part(part, &first_step, &last_step)) {
        fprintf(stderr, "Usage: gpt1_bigram_training [all|1|2|3|1-2|1-3]\n");
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
    size_t validation_length = token_count - train_length;
    int *validation_data = token_ids + train_length;
    int train_eval_inputs[BATCH_SIZE * BLOCK_SIZE];
    int train_eval_targets[BATCH_SIZE * BLOCK_SIZE];
    int validation_inputs[BATCH_SIZE * BLOCK_SIZE];
    int validation_targets[BATCH_SIZE * BLOCK_SIZE];
    int batch_inputs[BATCH_SIZE * BLOCK_SIZE];
    int batch_targets[BATCH_SIZE * BLOCK_SIZE];

    srand(1337);
    sample_batch(token_ids, train_length, BATCH_SIZE,
                 train_eval_inputs, train_eval_targets);
    sample_batch(validation_data, validation_length, BATCH_SIZE,
                 validation_inputs, validation_targets);

    BigramLanguageModel model = {0};
    if (!bigram_model_init(&model, vocabulary.size, 1337)) {
        fprintf(stderr, "Could not initialize the bigram model.\n");
        free(token_ids);
        return EXIT_FAILURE;
    }

    size_t logits_count = BATCH_SIZE * BLOCK_SIZE * vocabulary.size;
    float *logits = malloc(logits_count * sizeof(*logits));
    if (logits == NULL) {
        fprintf(stderr, "Could not allocate evaluation logits.\n");
        bigram_model_destroy(&model);
        free(token_ids);
        return EXIT_FAILURE;
    }

    if (first_step <= 1) {
        double train_loss = evaluate(&model, train_eval_inputs, train_eval_targets, logits);
        double validation_loss = evaluate(&model, validation_inputs,
                                         validation_targets, logits);
        printf("Step 1 - Establish the starting point\n");
        printf("Model parameters: %zu x %zu = %zu scores\n", vocabulary.size,
               vocabulary.size, vocabulary.size * vocabulary.size);
        printf("Evaluation batch shapes: x/y = [%zu, %zu]\n", BATCH_SIZE, BLOCK_SIZE);
        printf("Initial fixed-batch loss: train %.4f; validation %.4f\n",
               train_loss, validation_loss);
        puts("The same evaluation batches are reused to make before/after comparisons clearer.");
    }

    if (last_step >= 2) {
        const double learning_rate = 1e-3;
        const double weight_decay = 0.01;
        if (first_step <= 2) {
            puts("Step 2 - Train the table with AdamW for 100 mini-batches");
            puts("AdamW keeps moving averages of gradients and squared gradients.");
            puts("step | train loss | validation loss");
        }

        srand(2025);
        for (size_t step = 0; step <= TRAINING_STEPS; step++) {
            if (first_step <= 2 && step % 25 == 0) {
                double train_loss = evaluate(&model, train_eval_inputs,
                                             train_eval_targets, logits);
                double validation_loss = evaluate(&model, validation_inputs,
                                                  validation_targets, logits);
                print_loss_line(step, train_loss, validation_loss);
            }
            if (step == TRAINING_STEPS) break;

            sample_batch(token_ids, train_length, BATCH_SIZE, batch_inputs, batch_targets);
            if (!bigram_model_train_batch(&model, batch_inputs, batch_targets,
                                          BATCH_SIZE, BLOCK_SIZE,
                                          learning_rate, weight_decay)) {
                fprintf(stderr, "Training update failed at step %zu.\n", step + 1);
                free(logits);
                bigram_model_destroy(&model);
                free(token_ids);
                return EXIT_FAILURE;
            }
        }
    }

    if (last_step >= 3) {
        const size_t new_token_count = 200;
        int generated_ids[201];
        bigram_model_generate(&model, 0, new_token_count, generated_ids);
        char *generated_text = vocabulary_decode(generated_ids,
                                                 new_token_count + 1, &vocabulary);
        if (generated_text == NULL) {
            fprintf(stderr, "Could not decode generated token IDs.\n");
            free(logits);
            bigram_model_destroy(&model);
            free(token_ids);
            return EXIT_FAILURE;
        }
        puts("Step 3 - Sample text from the trained bigram model");
        printf("Start token ID: 0; generated text:\n%s\n", generated_text);
        puts("The model now prefers observed character pairs, but it only remembers one character.");
        free(generated_text);
    }

    free(logits);
    bigram_model_destroy(&model);
    free(token_ids);
    return EXIT_SUCCESS;
}
