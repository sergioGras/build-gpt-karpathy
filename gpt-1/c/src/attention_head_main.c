#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { BATCH_SIZE = 2, TIME_STEPS = 4, CHANNELS = 4, HEAD_SIZE = 2 };

static const double INPUT[BATCH_SIZE][TIME_STEPS][CHANNELS] = {
    {
        { 1.0, 0.0, 2.0, -1.0 }, { 0.0, 1.0, 1.0, 2.0 },
        { 2.0, -1.0, 0.0, 1.0 }, { 1.0, 2.0, -1.0, 0.0 }
    },
    {
        { 0.0, 2.0, 1.0, -1.0 }, { 1.0, -1.0, 2.0, 0.0 },
        { -1.0, 0.0, 1.0, 2.0 }, { 2.0, 1.0, 0.0, -1.0 }
    }
};

/* Stored as [output channel][input channel], like nn.Linear.weight. */
static const double QUERY_WEIGHT[HEAD_SIZE][CHANNELS] = {
    { 0.5, -0.5, 1.0, 0.0 }, { 0.0, 1.0, 0.5, -0.5 }
};
static const double KEY_WEIGHT[HEAD_SIZE][CHANNELS] = {
    { 1.0, 0.0, 0.5, -0.5 }, { -0.5, 0.5, 0.0, 1.0 }
};
static const double VALUE_WEIGHT[HEAD_SIZE][CHANNELS] = {
    { 0.5, 0.0, 1.0, 0.5 }, { 0.0, 1.0, -0.5, 0.5 }
};

typedef struct {
    double queries[BATCH_SIZE][TIME_STEPS][HEAD_SIZE];
    double keys[BATCH_SIZE][TIME_STEPS][HEAD_SIZE];
    double values[BATCH_SIZE][TIME_STEPS][HEAD_SIZE];
    double raw_scores[BATCH_SIZE][TIME_STEPS][TIME_STEPS];
    double scaled_scores[BATCH_SIZE][TIME_STEPS][TIME_STEPS];
    double masked_scores[BATCH_SIZE][TIME_STEPS][TIME_STEPS];
    double weights[BATCH_SIZE][TIME_STEPS][TIME_STEPS];
    double output[BATCH_SIZE][TIME_STEPS][HEAD_SIZE];
} AttentionResult;

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

static void project(const double input[BATCH_SIZE][TIME_STEPS][CHANNELS],
                    const double weight[HEAD_SIZE][CHANNELS],
                    double projected[BATCH_SIZE][TIME_STEPS][HEAD_SIZE]) {
    for (size_t batch = 0; batch < BATCH_SIZE; batch++) {
        for (size_t time = 0; time < TIME_STEPS; time++) {
            for (size_t output_channel = 0; output_channel < HEAD_SIZE; output_channel++) {
                projected[batch][time][output_channel] = 0.0;
                for (size_t input_channel = 0; input_channel < CHANNELS; input_channel++) {
                    projected[batch][time][output_channel] +=
                        input[batch][time][input_channel] * weight[output_channel][input_channel];
                }
            }
        }
    }
}

static void forward(const double input[BATCH_SIZE][TIME_STEPS][CHANNELS],
                    AttentionResult *result) {
    project(input, QUERY_WEIGHT, result->queries);
    project(input, KEY_WEIGHT, result->keys);
    project(input, VALUE_WEIGHT, result->values);

    for (size_t batch = 0; batch < BATCH_SIZE; batch++) {
        for (size_t time = 0; time < TIME_STEPS; time++) {
            for (size_t key_time = 0; key_time < TIME_STEPS; key_time++) {
                double score = 0.0;
                for (size_t channel = 0; channel < HEAD_SIZE; channel++) {
                    score += result->queries[batch][time][channel]
                           * result->keys[batch][key_time][channel];
                }
                result->raw_scores[batch][time][key_time] = score;
                result->scaled_scores[batch][time][key_time] = score / sqrt((double)HEAD_SIZE);
                result->masked_scores[batch][time][key_time] =
                    key_time <= time ? result->scaled_scores[batch][time][key_time] : -INFINITY;
            }
        }
    }

    for (size_t batch = 0; batch < BATCH_SIZE; batch++) {
        for (size_t time = 0; time < TIME_STEPS; time++) {
            double maximum = result->masked_scores[batch][time][0];
            for (size_t key_time = 1; key_time <= time; key_time++) {
                if (result->masked_scores[batch][time][key_time] > maximum) {
                    maximum = result->masked_scores[batch][time][key_time];
                }
            }

            double total = 0.0;
            for (size_t key_time = 0; key_time < TIME_STEPS; key_time++) {
                double score = result->masked_scores[batch][time][key_time];
                result->weights[batch][time][key_time] =
                    isinf(score) ? 0.0 : exp(score - maximum);
                total += result->weights[batch][time][key_time];
            }
            for (size_t key_time = 0; key_time < TIME_STEPS; key_time++) {
                result->weights[batch][time][key_time] /= total;
            }
        }
    }

    for (size_t batch = 0; batch < BATCH_SIZE; batch++) {
        for (size_t time = 0; time < TIME_STEPS; time++) {
            for (size_t channel = 0; channel < HEAD_SIZE; channel++) {
                result->output[batch][time][channel] = 0.0;
                for (size_t key_time = 0; key_time < TIME_STEPS; key_time++) {
                    result->output[batch][time][channel] +=
                        result->weights[batch][time][key_time]
                        * result->values[batch][key_time][channel];
                }
            }
        }
    }
}

static void print_vectors(const char *label,
                          const double values[BATCH_SIZE][TIME_STEPS][HEAD_SIZE]) {
    printf("%s, shape [%d, %d, %d]:\n", label, BATCH_SIZE, TIME_STEPS, HEAD_SIZE);
    for (size_t batch = 0; batch < BATCH_SIZE; batch++) {
        printf("  batch %zu\n", batch);
        for (size_t time = 0; time < TIME_STEPS; time++) {
            printf("    t=%zu [%.4f, %.4f]\n", time,
                   values[batch][time][0], values[batch][time][1]);
        }
    }
}

static void print_scores(const char *label,
                         const double values[BATCH_SIZE][TIME_STEPS][TIME_STEPS]) {
    printf("%s, shape [%d, %d, %d] (batch 0 shown):\n",
           label, BATCH_SIZE, TIME_STEPS, TIME_STEPS);
    for (size_t time = 0; time < TIME_STEPS; time++) {
        putchar(' ');
        for (size_t key_time = 0; key_time < TIME_STEPS; key_time++) {
            if (isinf(values[0][time][key_time])) printf(" %8s", "-inf");
            else printf(" %8.4f", values[0][time][key_time]);
        }
        putchar('\n');
    }
}

static double maximum_prefix_difference(
    const double left[BATCH_SIZE][TIME_STEPS][HEAD_SIZE],
    const double right[BATCH_SIZE][TIME_STEPS][HEAD_SIZE],
    size_t batch, size_t prefix_length) {
    double maximum = 0.0;
    for (size_t time = 0; time < prefix_length; time++) {
        for (size_t channel = 0; channel < HEAD_SIZE; channel++) {
            double difference = fabs(left[batch][time][channel] - right[batch][time][channel]);
            if (difference > maximum) maximum = difference;
        }
    }
    return maximum;
}

int main(int argc, char *argv[]) {
    const char *part = argc == 1 ? "all" : argv[1];
    int first_step;
    int last_step;
    if (argc > 2 || !parse_part(part, &first_step, &last_step)) {
        fprintf(stderr, "Usage: gpt1_attention_head [all|1|2|3|4|1-2|1-3|1-4|2-4|3-4]\n");
        return EXIT_FAILURE;
    }

    AttentionResult result;
    forward(INPUT, &result);

    if (first_step <= 1) {
        puts("Step 1 - Project each token into query, key, and value vectors");
        puts("Input shape [batch, time, channels] = [2, 4, 4]; each projection has head size 2.");
        puts("These fixed matrices stand in for the trainable weights of three nn.Linear layers.");
        print_vectors("Queries", result.queries);
        print_vectors("Keys", result.keys);
        print_vectors("Values", result.values);
    }

    if (first_step <= 2 && last_step >= 2) {
        puts("Step 2 - Compare each query with every key and scale the dot products");
        print_scores("Raw query-key scores", result.raw_scores);
        print_scores("Scaled scores = query dot key / sqrt(head_size)", result.scaled_scores);
        puts("Scores have shape [batch, query time, key time] = [2, 4, 4].");
    }

    if (first_step <= 3 && last_step >= 3) {
        puts("Step 3 - Hide future positions, then normalize each row with softmax");
        print_scores("Scores after causal masking", result.masked_scores);
        print_scores("Attention weights", result.weights);
        puts("Every row sums to 1; future positions have weight 0.");
    }

    if (last_step >= 4) {
        puts("Step 4 - Use attention weights to combine the value vectors");
        print_vectors("Single-head output", result.output);
        puts("The output shape is [batch, time, head_size] = [2, 4, 2].");

        double changed_input[BATCH_SIZE][TIME_STEPS][CHANNELS];
        memcpy(changed_input, INPUT, sizeof(changed_input));
        changed_input[0][3][0] += 100.0;
        AttentionResult changed_result;
        forward(changed_input, &changed_result);

        double earlier_difference =
            maximum_prefix_difference(result.output, changed_result.output, 0, 3);
        double final_difference = 0.0;
        for (size_t channel = 0; channel < HEAD_SIZE; channel++) {
            double difference = fabs(result.output[0][3][channel]
                                   - changed_result.output[0][3][channel]);
            if (difference > final_difference) final_difference = difference;
        }
        printf("Change a future input token by +100: max change at earlier positions = %.12f\n",
               earlier_difference);
        printf("Change at the edited position's output = %.12f\n", final_difference);
        if (earlier_difference > 1e-10 || final_difference < 1e-8) {
            fputs("Causality check failed.\n", stderr);
            return EXIT_FAILURE;
        }
        puts("Causality check passed: future input changes cannot alter earlier outputs.");
    }

    return EXIT_SUCCESS;
}
