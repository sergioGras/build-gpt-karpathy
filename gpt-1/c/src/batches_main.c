#include "dataset.h"
#include "vocabulary.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int parse_part(const char *part, int *first_step, int *last_step) {
    *first_step = 1;
    *last_step = 3;

    if (strcmp(part, "1") == 0) {
        *last_step = 1;
    } else if (strcmp(part, "2") == 0) {
        *first_step = *last_step = 2;
    } else if (strcmp(part, "3") == 0) {
        *first_step = *last_step = 3;
    } else if (strcmp(part, "1-2") == 0) {
        *last_step = 2;
    } else if (strcmp(part, "all") != 0 && strcmp(part, "1-3") != 0) {
        return 0;
    }
    return 1;
}

static void print_ids(const int *token_ids, size_t token_count) {
    for (size_t index = 0; index < token_count; index++) {
        printf("%s%d", index == 0 ? "" : " ", token_ids[index]);
    }
    putchar('\n');
}

static void print_character(unsigned char character) {
    if (character == '\n') {
        fputs("'\\n'", stdout);
    } else if (character == '\t') {
        fputs("'\\t'", stdout);
    } else {
        printf("'%c'", character);
    }
}

static void print_escaped_text(const char *text) {
    putchar('"');
    for (const unsigned char *character = (const unsigned char *)text;
         *character != '\0'; character++) {
        if (*character == '\n') {
            fputs("\\n", stdout);
        } else if (*character == '\t') {
            fputs("\\t", stdout);
        } else if (*character == '\\' || *character == '"') {
            putchar('\\');
            putchar(*character);
        } else {
            putchar(*character);
        }
    }
    putchar('"');
}

static void sample_batch(const int *data, size_t data_length, size_t batch_size,
                         size_t block_size, int *inputs, int *targets) {
    size_t start_choices = data_length - block_size;

    for (size_t batch_index = 0; batch_index < batch_size; batch_index++) {
        size_t start = (size_t)rand() % start_choices;
        for (size_t time = 0; time < block_size; time++) {
            size_t output_index = batch_index * block_size + time;
            inputs[output_index] = data[start + time];
            targets[output_index] = data[start + time + 1];
        }
    }
}

static void print_matrix(const char *name, const int *values,
                         size_t batch_size, size_t block_size) {
    printf("%s, shape [%zu, %zu]:\n", name, batch_size, block_size);
    for (size_t batch_index = 0; batch_index < batch_size; batch_index++) {
        print_ids(values + batch_index * block_size, block_size);
    }
}

int main(int argc, char *argv[]) {
    const char *part = argc == 1 ? "all" : argv[1];
    int first_step;
    int last_step;

    if (argc > 2 || !parse_part(part, &first_step, &last_step)) {
        fprintf(stderr, "Usage: gpt1_batches [all|1|2|3|1-2|1-3]\n");
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

    size_t split_index = token_count * 9 / 10;
    size_t train_length = split_index;
    size_t validation_length = token_count - split_index;
    const size_t block_size = 8;

    if (first_step <= 1) {
        puts("Step 1 - Split tokens into training and validation sets");
        printf("All tokens: %zu\n", token_count);
        printf("Training tokens (first 90%%): %zu\n", train_length);
        printf("Validation tokens (last 10%%): %zu\n", validation_length);
    }

    if (first_step <= 2 && last_step >= 2) {
        if (train_length <= block_size) {
            fprintf(stderr, "Training split is too short for block size %zu.\n", block_size);
            free(token_ids);
            return EXIT_FAILURE;
        }

        puts("Step 2 - Make context and next-token examples");
        printf("Block size: %zu; the displayed window has %zu tokens.\n", block_size, block_size + 1);
        printf("Window token IDs: ");
        print_ids(token_ids, block_size + 1);
        for (size_t time = 0; time < block_size; time++) {
            char *context_text = vocabulary_decode(token_ids, time + 1, &vocabulary);
            if (context_text == NULL) {
                fprintf(stderr, "Could not decode the context window.\n");
                free(token_ids);
                return EXIT_FAILURE;
            }
            printf("Context IDs ");
            print_ids(token_ids, time + 1);
            printf("  (");
            print_escaped_text(context_text);
            printf(") -> target ID %d, character ", token_ids[time + 1]);
            print_character(vocabulary.id_to_byte[token_ids[time + 1]]);
            putchar('\n');
            free(context_text);
        }
    }

    if (last_step >= 3) {
        const size_t batch_size = 4;
        int *inputs = malloc(batch_size * block_size * sizeof(*inputs));
        int *targets = malloc(batch_size * block_size * sizeof(*targets));
        if (inputs == NULL || targets == NULL) {
            fprintf(stderr, "Could not allocate the example batch.\n");
            free(inputs);
            free(targets);
            free(token_ids);
            return EXIT_FAILURE;
        }

        /* C's rand() is seeded for repeatability; its sequence differs from PyTorch/Java. */
        srand(1337);
        sample_batch(token_ids, train_length, batch_size, block_size, inputs, targets);

        puts("Step 3 - Sample a mini-batch");
        printf("Each row is one randomly selected training window; each row has %zu positions.\n",
               block_size);
        print_matrix("Training x inputs", inputs, batch_size, block_size);
        print_matrix("Training y next-token targets", targets, batch_size, block_size);

        sample_batch(token_ids + split_index, validation_length,
                     batch_size, block_size, inputs, targets);
        printf("Validation batch shapes are also [%zu, %zu].\n", batch_size, block_size);
        print_matrix("Validation x inputs", inputs, batch_size, block_size);
        print_matrix("Validation y next-token targets", targets, batch_size, block_size);

        free(inputs);
        free(targets);
    }

    free(token_ids);
    return EXIT_SUCCESS;
}
