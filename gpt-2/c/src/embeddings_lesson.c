#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { BATCH_SIZE = 2, SEQUENCE_LENGTH = 3, EMBEDDING_SIZE = 2, VOCAB_SIZE = 4 };

/* Tiny learned-parameter stand-ins. The GPT-2 tables are much larger. */
static const int token_ids[BATCH_SIZE][SEQUENCE_LENGTH] = {
    {1, 2, 3},
    {3, 2, 1}
};
static const double token_embedding_table[VOCAB_SIZE][EMBEDDING_SIZE] = {
    {0.0, 0.0},
    {1.0, 0.0},
    {0.0, 1.0},
    {1.0, 1.0}
};
static const double position_embedding_table[SEQUENCE_LENGTH][EMBEDDING_SIZE] = {
    {0.1, 0.0},
    {0.0, 0.2},
    {0.3, 0.3}
};

static void build_token_vectors(double output[BATCH_SIZE][SEQUENCE_LENGTH][EMBEDDING_SIZE]) {
    for (size_t batch = 0; batch < BATCH_SIZE; ++batch) {
        for (size_t position = 0; position < SEQUENCE_LENGTH; ++position) {
            const int token_id = token_ids[batch][position];
            for (size_t channel = 0; channel < EMBEDDING_SIZE; ++channel) {
                output[batch][position][channel] = token_embedding_table[token_id][channel];
            }
        }
    }
}

static void print_token_ids(void) {
    puts("Token IDs, shape (B, T) = (2, 3):");
    for (size_t batch = 0; batch < BATCH_SIZE; ++batch) {
        printf("  [%d, %d, %d]\n", token_ids[batch][0], token_ids[batch][1], token_ids[batch][2]);
    }
    puts("Position IDs, shape (T): [0, 1, 2]");
}

static void print_vectors(const char *label,
        double values[BATCH_SIZE][SEQUENCE_LENGTH][EMBEDDING_SIZE]) {
    printf("%s, shape (B, T, C) = (2, 3, 2):\n", label);
    for (size_t batch = 0; batch < BATCH_SIZE; ++batch) {
        printf("  batch %zu: ", batch);
        for (size_t position = 0; position < SEQUENCE_LENGTH; ++position) {
            printf("[%0.1f, %0.1f]%s", values[batch][position][0],
                values[batch][position][1], position + 1 == SEQUENCE_LENGTH ? "\n" : " ");
        }
    }
}

static bool show_part(const char *part) {
    double token_vectors[BATCH_SIZE][SEQUENCE_LENGTH][EMBEDDING_SIZE];
    double input_vectors[BATCH_SIZE][SEQUENCE_LENGTH][EMBEDDING_SIZE];
    build_token_vectors(token_vectors);

    if (strcmp(part, "1") == 0 || strcmp(part, "all") == 0) {
        puts("Step 1: token IDs and positions are separate inputs");
        print_token_ids();
        puts("GPT-2 maps each token ID to a 768-number vector and supports up to 1,024 positions.\n");
    }
    if (strcmp(part, "2") == 0 || strcmp(part, "all") == 0) {
        puts("Step 2: look up one learned vector for each token ID");
        print_vectors("Token vectors", token_vectors);
        puts("The table lookup changes integer IDs into continuous representations.\n");
    }
    if (strcmp(part, "3") == 0 || strcmp(part, "all") == 0) {
        puts("Step 3: add the position vector at each sequence location");
        for (size_t batch = 0; batch < BATCH_SIZE; ++batch) {
            for (size_t position = 0; position < SEQUENCE_LENGTH; ++position) {
                for (size_t channel = 0; channel < EMBEDDING_SIZE; ++channel) {
                    input_vectors[batch][position][channel] = token_vectors[batch][position][channel]
                        + position_embedding_table[position][channel];
                }
            }
        }
        print_vectors("Token + position vectors (initial residual stream)", input_vectors);
        puts("The position table is reused across rows of the batch; broadcasting is explicit in these loops.\n");
    }
    return true;
}

int main(int argc, char **argv) {
    const char *part = argc == 2 ? argv[1] : "all";
    if (argc > 2 || (strcmp(part, "all") != 0 && strcmp(part, "1") != 0
            && strcmp(part, "2") != 0 && strcmp(part, "3") != 0)) {
        fprintf(stderr, "Usage: embeddings_lesson [all|1|2|3]\n");
        return EXIT_FAILURE;
    }
    return show_part(part) ? EXIT_SUCCESS : EXIT_FAILURE;
}
