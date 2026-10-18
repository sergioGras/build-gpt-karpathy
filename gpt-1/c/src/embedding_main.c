#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { BATCH_SIZE = 2, TIME_STEPS = 4, CHANNELS = 4, VOCAB_SIZE = 5 };

static const char *const VOCABULARY[VOCAB_SIZE] = { "a", "b", "c", "d", "e" };
static const int TOKEN_IDS[BATCH_SIZE][TIME_STEPS] = {
    { 0, 1, 0, 2 },
    { 1, 2, 3, 0 }
};

static int parse_part(const char *part, int *first_step, int *last_step) {
    *first_step = 1;
    *last_step = 4;
    if (strcmp(part, "1") == 0) *last_step = 1;
    else if (strcmp(part, "2") == 0) *first_step = *last_step = 2;
    else if (strcmp(part, "3") == 0) *first_step = *last_step = 3;
    else if (strcmp(part, "4") == 0) *first_step = *last_step = 4;
    else if (strcmp(part, "1-2") == 0) *last_step = 2;
    else if (strcmp(part, "1-3") == 0) *last_step = 3;
    else if (strcmp(part, "2-4") == 0) *first_step = 2;
    else if (strcmp(part, "3-4") == 0) *first_step = 3;
    else if (strcmp(part, "1-4") != 0 && strcmp(part, "all") != 0) return 0;
    return 1;
}

static void initialize_embedding_tables(
    double token_table[VOCAB_SIZE][CHANNELS],
    double position_table[TIME_STEPS][CHANNELS]) {
    for (size_t token = 0; token < VOCAB_SIZE; token++) {
        for (size_t channel = 0; channel < CHANNELS; channel++) {
            int pattern = (int)((token * 5 + channel * 3 + 1) % 11) - 5;
            token_table[token][channel] = pattern * 0.1;
        }
    }
    for (size_t position = 0; position < TIME_STEPS; position++) {
        for (size_t channel = 0; channel < CHANNELS; channel++) {
            int pattern = (int)((position * 7 + channel * 2 + 2) % 9) - 4;
            position_table[position][channel] = pattern * 0.1;
        }
    }
}

static void lookup_embeddings(
    const double token_table[VOCAB_SIZE][CHANNELS],
    const double position_table[TIME_STEPS][CHANNELS],
    double token_vectors[BATCH_SIZE][TIME_STEPS][CHANNELS],
    double position_vectors[BATCH_SIZE][TIME_STEPS][CHANNELS],
    double model_input[BATCH_SIZE][TIME_STEPS][CHANNELS]) {
    for (size_t batch = 0; batch < BATCH_SIZE; batch++) {
        for (size_t time = 0; time < TIME_STEPS; time++) {
            int token_id = TOKEN_IDS[batch][time];
            for (size_t channel = 0; channel < CHANNELS; channel++) {
                token_vectors[batch][time][channel] = token_table[token_id][channel];
                position_vectors[batch][time][channel] = position_table[time][channel];
                model_input[batch][time][channel] =
                    token_vectors[batch][time][channel] + position_vectors[batch][time][channel];
            }
        }
    }
}

static void print_ids(void) {
    puts("Vocabulary: a=0, b=1, c=2, d=3, e=4");
    printf("Token IDs, shape [batch, time] = [%d, %d]:\n", BATCH_SIZE, TIME_STEPS);
    for (size_t batch = 0; batch < BATCH_SIZE; batch++) {
        printf("  batch %zu [", batch);
        for (size_t time = 0; time < TIME_STEPS; time++) {
            printf("%s%d", time == 0 ? "" : ", ", TOKEN_IDS[batch][time]);
        }
        puts("]");
    }
}

static void print_token_vectors(
    const char *label, const double values[BATCH_SIZE][TIME_STEPS][CHANNELS]) {
    printf("%s, shape [batch, time, channels] = [%d, %d, %d]:\n",
           label, BATCH_SIZE, TIME_STEPS, CHANNELS);
    for (size_t batch = 0; batch < BATCH_SIZE; batch++) {
        printf("  batch %zu\n", batch);
        for (size_t time = 0; time < TIME_STEPS; time++) {
            printf("    token %s at position %zu [",
                   VOCABULARY[TOKEN_IDS[batch][time]], time);
            for (size_t channel = 0; channel < CHANNELS; channel++) {
                printf("%s%.2f", channel == 0 ? "" : ", ", values[batch][time][channel]);
            }
            puts("]");
        }
    }
}

static double maximum_difference(const double left[CHANNELS], const double right[CHANNELS]) {
    double maximum = 0.0;
    for (size_t channel = 0; channel < CHANNELS; channel++) {
        double difference = fabs(left[channel] - right[channel]);
        if (difference > maximum) maximum = difference;
    }
    return maximum;
}

int main(int argc, char *argv[]) {
    const char *part = argc == 1 ? "all" : argv[1];
    int first_step;
    int last_step;
    if (argc > 2 || !parse_part(part, &first_step, &last_step)) {
        fprintf(stderr, "Usage: gpt1_embeddings [all|1|2|3|4|1-2|1-3|1-4|2-4|3-4]\n");
        return EXIT_FAILURE;
    }

    double token_table[VOCAB_SIZE][CHANNELS];
    double position_table[TIME_STEPS][CHANNELS];
    double token_vectors[BATCH_SIZE][TIME_STEPS][CHANNELS];
    double position_vectors[BATCH_SIZE][TIME_STEPS][CHANNELS];
    double model_input[BATCH_SIZE][TIME_STEPS][CHANNELS];
    initialize_embedding_tables(token_table, position_table);
    lookup_embeddings(token_table, position_table, token_vectors, position_vectors, model_input);

    if (first_step <= 1) {
        puts("Step 1 - Start with token IDs");
        print_ids();
        puts("Each integer is an index into a vocabulary; the number itself has no ordered meaning.");
    }

    if (first_step <= 2 && last_step >= 2) {
        puts("Step 2 - Look up a learned vector for each token");
        print_token_vectors("Token embeddings", token_vectors);
        puts("The same token ID always selects the same row from the token embedding table.");
    }

    if (first_step <= 3 && last_step >= 3) {
        puts("Step 3 - Look up a learned vector for each position");
        printf("Position embedding table, shape [time, channels] = [%d, %d]:\n",
               TIME_STEPS, CHANNELS);
        for (size_t time = 0; time < TIME_STEPS; time++) {
            printf("  position %zu [", time);
            for (size_t channel = 0; channel < CHANNELS; channel++) {
                printf("%s%.2f", channel == 0 ? "" : ", ", position_table[time][channel]);
            }
            puts("]");
        }
        puts("The position table is added to every sequence in the batch.");
    }

    if (last_step >= 4) {
        double same_token_difference = maximum_difference(
            token_vectors[0][0], token_vectors[0][2]);
        double positioned_difference = maximum_difference(
            model_input[0][0], model_input[0][2]);

        puts("Step 4 - Add token and position information");
        print_token_vectors("Transformer input = token embedding + position embedding", model_input);
        printf("Same token at positions 0 and 2: token-vector difference = %.6f; combined-vector difference = %.6f\n",
               same_token_difference, positioned_difference);
        if (same_token_difference > 1e-12 || positioned_difference < 1e-12) {
            fputs("Embedding position check failed.\n", stderr);
            return EXIT_FAILURE;
        }
        puts("Check passed: the token vector is reused, while its position changes the combined representation.");
    }

    return EXIT_SUCCESS;
}
