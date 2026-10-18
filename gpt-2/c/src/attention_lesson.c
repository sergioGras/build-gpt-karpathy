#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { SEQUENCE_LENGTH = 3, EMBEDDING_SIZE = 2 };

/* This is the first sequence's residual stream from Lesson 4. */
static double input[SEQUENCE_LENGTH][EMBEDDING_SIZE] = {
    {1.1, 0.0},
    {0.0, 1.2},
    {1.3, 1.3}
};

/* Identity projections keep this first example focused on attention weights. */
static double query_weights[EMBEDDING_SIZE][EMBEDDING_SIZE] = {
    {1.0, 0.0}, {0.0, 1.0}
};
static double key_weights[EMBEDDING_SIZE][EMBEDDING_SIZE] = {
    {1.0, 0.0}, {0.0, 1.0}
};
static double value_weights[EMBEDDING_SIZE][EMBEDDING_SIZE] = {
    {1.0, 0.0}, {0.0, 1.0}
};

static void project(double values[SEQUENCE_LENGTH][EMBEDDING_SIZE],
        double weights[EMBEDDING_SIZE][EMBEDDING_SIZE],
        double projected[SEQUENCE_LENGTH][EMBEDDING_SIZE]) {
    for (size_t position = 0; position < SEQUENCE_LENGTH; ++position) {
        for (size_t output_channel = 0; output_channel < EMBEDDING_SIZE; ++output_channel) {
            projected[position][output_channel] = 0.0;
            for (size_t input_channel = 0; input_channel < EMBEDDING_SIZE; ++input_channel) {
                projected[position][output_channel] += values[position][input_channel]
                    * weights[input_channel][output_channel];
            }
        }
    }
}

static void build_scores(double query[SEQUENCE_LENGTH][EMBEDDING_SIZE],
        double key[SEQUENCE_LENGTH][EMBEDDING_SIZE],
        double scores[SEQUENCE_LENGTH][SEQUENCE_LENGTH]) {
    const double scale = sqrt((double) EMBEDDING_SIZE);
    for (size_t query_position = 0; query_position < SEQUENCE_LENGTH; ++query_position) {
        for (size_t key_position = 0; key_position < SEQUENCE_LENGTH; ++key_position) {
            double dot_product = 0.0;
            for (size_t channel = 0; channel < EMBEDDING_SIZE; ++channel) {
                dot_product += query[query_position][channel] * key[key_position][channel];
            }
            scores[query_position][key_position] = dot_product / scale;
        }
    }
}

static void apply_causal_mask(double scores[SEQUENCE_LENGTH][SEQUENCE_LENGTH],
        double masked_scores[SEQUENCE_LENGTH][SEQUENCE_LENGTH]) {
    for (size_t query_position = 0; query_position < SEQUENCE_LENGTH; ++query_position) {
        for (size_t key_position = 0; key_position < SEQUENCE_LENGTH; ++key_position) {
            /* A token may use itself and earlier positions, never future positions. */
            masked_scores[query_position][key_position] = key_position > query_position
                ? -INFINITY : scores[query_position][key_position];
        }
    }
}

static void softmax_rows(double logits[SEQUENCE_LENGTH][SEQUENCE_LENGTH],
        double weights[SEQUENCE_LENGTH][SEQUENCE_LENGTH]) {
    for (size_t query_position = 0; query_position < SEQUENCE_LENGTH; ++query_position) {
        double maximum = -INFINITY;
        for (size_t key_position = 0; key_position < SEQUENCE_LENGTH; ++key_position) {
            if (logits[query_position][key_position] > maximum) {
                maximum = logits[query_position][key_position];
            }
        }

        double denominator = 0.0;
        for (size_t key_position = 0; key_position < SEQUENCE_LENGTH; ++key_position) {
            weights[query_position][key_position] = exp(logits[query_position][key_position] - maximum);
            denominator += weights[query_position][key_position];
        }
        for (size_t key_position = 0; key_position < SEQUENCE_LENGTH; ++key_position) {
            weights[query_position][key_position] /= denominator;
        }
    }
}

static void combine_values(double weights[SEQUENCE_LENGTH][SEQUENCE_LENGTH],
        double value[SEQUENCE_LENGTH][EMBEDDING_SIZE],
        double context[SEQUENCE_LENGTH][EMBEDDING_SIZE]) {
    for (size_t query_position = 0; query_position < SEQUENCE_LENGTH; ++query_position) {
        for (size_t channel = 0; channel < EMBEDDING_SIZE; ++channel) {
            context[query_position][channel] = 0.0;
            for (size_t key_position = 0; key_position < SEQUENCE_LENGTH; ++key_position) {
                context[query_position][channel] += weights[query_position][key_position]
                    * value[key_position][channel];
            }
        }
    }
}

static void run_attention(double values[SEQUENCE_LENGTH][EMBEDDING_SIZE],
        double query[SEQUENCE_LENGTH][EMBEDDING_SIZE],
        double key[SEQUENCE_LENGTH][EMBEDDING_SIZE],
        double value[SEQUENCE_LENGTH][EMBEDDING_SIZE],
        double scores[SEQUENCE_LENGTH][SEQUENCE_LENGTH],
        double masked_scores[SEQUENCE_LENGTH][SEQUENCE_LENGTH],
        double weights[SEQUENCE_LENGTH][SEQUENCE_LENGTH],
        double context[SEQUENCE_LENGTH][EMBEDDING_SIZE]) {
    project(values, query_weights, query);
    project(values, key_weights, key);
    project(values, value_weights, value);
    build_scores(query, key, scores);
    apply_causal_mask(scores, masked_scores);
    softmax_rows(masked_scores, weights);
    combine_values(weights, value, context);
}

static void print_vector_matrix(const char *label,
        double values[SEQUENCE_LENGTH][EMBEDDING_SIZE]) {
    printf("%s (%d x %d):\n", label, SEQUENCE_LENGTH, EMBEDDING_SIZE);
    for (size_t row = 0; row < SEQUENCE_LENGTH; ++row) {
        printf("  [");
        for (size_t column = 0; column < EMBEDDING_SIZE; ++column) {
            printf("%s%0.4f", column == 0 ? "" : ", ", values[row][column]);
        }
        puts("]");
    }
}

static void print_score_matrix(const char *label,
        double values[SEQUENCE_LENGTH][SEQUENCE_LENGTH]) {
    printf("%s (%d x %d):\n", label, SEQUENCE_LENGTH, SEQUENCE_LENGTH);
    for (size_t row = 0; row < SEQUENCE_LENGTH; ++row) {
        printf("  [");
        for (size_t column = 0; column < SEQUENCE_LENGTH; ++column) {
            printf("%s%0.4f", column == 0 ? "" : ", ", values[row][column]);
        }
        puts("]");
    }
}

static bool selected(const char *part, const char *step) {
    return strcmp(part, "all") == 0 || strcmp(part, step) == 0;
}

int main(int argc, char **argv) {
    const char *part = argc == 2 ? argv[1] : "all";
    if (argc > 2 || (strcmp(part, "all") != 0 && strcmp(part, "1") != 0
            && strcmp(part, "2") != 0 && strcmp(part, "3") != 0)) {
        fprintf(stderr, "Usage: attention_lesson [all|1|2|3]\n");
        return EXIT_FAILURE;
    }

    double query[SEQUENCE_LENGTH][EMBEDDING_SIZE];
    double key[SEQUENCE_LENGTH][EMBEDDING_SIZE];
    double value[SEQUENCE_LENGTH][EMBEDDING_SIZE];
    double scores[SEQUENCE_LENGTH][SEQUENCE_LENGTH];
    double masked_scores[SEQUENCE_LENGTH][SEQUENCE_LENGTH];
    double weights[SEQUENCE_LENGTH][SEQUENCE_LENGTH];
    double context[SEQUENCE_LENGTH][EMBEDDING_SIZE];
    run_attention(input, query, key, value, scores, masked_scores, weights, context);

    if (selected(part, "1")) {
        puts("Step 1: project each input vector into query, key, and value");
        print_vector_matrix("Input residual stream X (T, C)", input);
        print_vector_matrix("Query Q = X W_Q", query);
        print_vector_matrix("Key K = X W_K", key);
        print_vector_matrix("Value V = X W_V", value);
        puts("The toy projections are identity matrices; GPT-2 learns separate projection weights.\n");
    }
    if (selected(part, "2")) {
        puts("Step 2: compare queries with keys, scale, then mask future positions");
        print_score_matrix("Scaled scores Q K^T / sqrt(2)", scores);
        print_score_matrix("After causal mask (future scores become -inf)", masked_scores);
        puts("Softmax turns each allowed score row into attention weights:\n");
        print_score_matrix("Attention weights", weights);
        puts("Each row sums to 1; masked future positions have weight 0.\n");
    }
    if (selected(part, "3")) {
        puts("Step 3: use the weights to combine value vectors");
        print_vector_matrix("Context vectors = attention weights x V", context);

        double changed_input[SEQUENCE_LENGTH][EMBEDDING_SIZE];
        memcpy(changed_input, input, sizeof(input));
        changed_input[2][0] += 100.0;
        changed_input[2][1] -= 100.0;
        double changed_query[SEQUENCE_LENGTH][EMBEDDING_SIZE];
        double changed_key[SEQUENCE_LENGTH][EMBEDDING_SIZE];
        double changed_value[SEQUENCE_LENGTH][EMBEDDING_SIZE];
        double changed_scores[SEQUENCE_LENGTH][SEQUENCE_LENGTH];
        double changed_masked[SEQUENCE_LENGTH][SEQUENCE_LENGTH];
        double changed_weights[SEQUENCE_LENGTH][SEQUENCE_LENGTH];
        double changed_context[SEQUENCE_LENGTH][EMBEDDING_SIZE];
        run_attention(changed_input, changed_query, changed_key, changed_value, changed_scores,
            changed_masked, changed_weights, changed_context);
        const bool earlier_output_unchanged = fabs(context[0][0] - changed_context[0][0]) < 1e-12
            && fabs(context[0][1] - changed_context[0][1]) < 1e-12;
        printf("Changing the final input leaves position 0's output unchanged: %s\n",
            earlier_output_unchanged ? "yes" : "no");
        if (!earlier_output_unchanged) {
            return EXIT_FAILURE;
        }
        puts("A decoder token can use its past and present, but cannot see the future.\n");
    }
    return EXIT_SUCCESS;
}
