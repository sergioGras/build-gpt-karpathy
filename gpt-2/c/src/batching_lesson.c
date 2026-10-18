#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { BATCH_SIZE = 2, CONTEXT_LENGTH = 4, WINDOW_SIZE = BATCH_SIZE * CONTEXT_LENGTH };

/* These are readable stand-in token IDs. Lesson 2 showed how text becomes IDs. */
static const int train_tokens[] = {101, 102, 103, 104, 105, 106, 107, 108, 109};
static const int validation_tokens[] = {201, 202, 203, 204, 205, 206, 207, 208, 209};

static void print_stream(const char *label, const int *tokens, size_t length) {
    printf("%s stream (%zu IDs): [", label, length);
    for (size_t i = 0; i < length; ++i) {
        printf("%s%d", i == 0 ? "" : ", ", tokens[i]);
    }
    puts("]");
}

static void print_matrix(const char *label, int values[BATCH_SIZE][CONTEXT_LENGTH]) {
    printf("%s =\n", label);
    for (size_t row = 0; row < BATCH_SIZE; ++row) {
        printf("  [");
        for (size_t column = 0; column < CONTEXT_LENGTH; ++column) {
            printf("%s%d", column == 0 ? "" : ", ", values[row][column]);
        }
        puts("]");
    }
}

static bool make_batch(const int *tokens, size_t token_count, size_t start,
        int inputs[BATCH_SIZE][CONTEXT_LENGTH],
        int targets[BATCH_SIZE][CONTEXT_LENGTH]) {
    if (start + WINDOW_SIZE + 1 > token_count) {
        return false;
    }

    for (size_t row = 0; row < BATCH_SIZE; ++row) {
        for (size_t column = 0; column < CONTEXT_LENGTH; ++column) {
            const size_t position = start + row * CONTEXT_LENGTH + column;
            inputs[row][column] = tokens[position];
            targets[row][column] = tokens[position + 1];
        }
    }
    return true;
}

static bool show_part(const char *part) {
    const size_t train_count = sizeof(train_tokens) / sizeof(train_tokens[0]);
    const size_t validation_count = sizeof(validation_tokens) / sizeof(validation_tokens[0]);
    if (strcmp(part, "1") == 0 || strcmp(part, "all") == 0) {
        puts("Step 1: keep training and validation token streams separate");
        print_stream("Train", train_tokens, train_count);
        print_stream("Validation", validation_tokens, validation_count);
        puts("The batch function receives one stream at a time, so a batch cannot cross the split.\n");
    }
    if (strcmp(part, "2") == 0 || strcmp(part, "all") == 0) {
        puts("Step 2: shift one stream by one token to make inputs and next-token targets");
        int inputs[BATCH_SIZE][CONTEXT_LENGTH];
        int targets[BATCH_SIZE][CONTEXT_LENGTH];
        if (!make_batch(train_tokens, train_count, 0, inputs, targets)) {
            fputs("The training stream is too short for one batch.\n", stderr);
            return false;
        }
        print_matrix("x (inputs)", inputs);
        print_matrix("y (targets)", targets);
        puts("Each y position is the next ID after its matching x position.\n");
    }
    if (strcmp(part, "3") == 0 || strcmp(part, "all") == 0) {
        puts("Step 3: advance the stream cursor and wrap when a full window will not fit");
        size_t cursor = WINDOW_SIZE;
        printf("After one batch, cursor = %zu.\n", cursor);
        if (cursor + WINDOW_SIZE + 1 > train_count) {
            cursor = 0;
            puts("The next B*T+1 window would overrun this toy stream, so restart at position 0.");
        }
        printf("Next batch starts at token position %zu.\n", cursor);
        puts("Karpathy's loader applies the same rule per shard and advances by B*T*num_processes.\n");
    }
    return true;
}

int main(int argc, char **argv) {
    const char *part = argc == 2 ? argv[1] : "all";
    if (argc > 2 || (strcmp(part, "all") != 0 && strcmp(part, "1") != 0
            && strcmp(part, "2") != 0 && strcmp(part, "3") != 0)) {
        fprintf(stderr, "Usage: batching_lesson [all|1|2|3]\n");
        return EXIT_FAILURE;
    }
    return show_part(part) ? EXIT_SUCCESS : EXIT_FAILURE;
}
