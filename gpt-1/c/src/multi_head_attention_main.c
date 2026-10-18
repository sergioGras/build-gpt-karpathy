#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { BATCH_SIZE = 2, TIME_STEPS = 4, CHANNELS = 4, HEADS = 2, HEAD_SIZE = 2 };

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

/* Weights are [head][output channel][input channel], like nn.Linear.weight. */
static const double QUERY_WEIGHT[HEADS][HEAD_SIZE][CHANNELS] = {
    {
        { 0.5, -0.5, 1.0, 0.0 }, { 0.0, 1.0, 0.5, -0.5 }
    },
    {
        { 0.0, 0.5, -0.5, 1.0 }, { 1.0, -0.5, 0.0, 0.5 }
    }
};
static const double KEY_WEIGHT[HEADS][HEAD_SIZE][CHANNELS] = {
    {
        { 1.0, 0.0, 0.5, -0.5 }, { -0.5, 0.5, 0.0, 1.0 }
    },
    {
        { 0.5, 1.0, -0.5, 0.0 }, { -1.0, 0.0, 0.5, 0.5 }
    }
};
static const double VALUE_WEIGHT[HEADS][HEAD_SIZE][CHANNELS] = {
    {
        { 0.5, 0.0, 1.0, 0.5 }, { 0.0, 1.0, -0.5, 0.5 }
    },
    {
        { 1.0, 0.5, 0.0, -0.5 }, { 0.5, -1.0, 1.0, 0.0 }
    }
};

static const double OUTPUT_WEIGHT[CHANNELS][CHANNELS] = {
    { 0.5, 0.0, 0.5, 0.0 },
    { 0.0, 0.5, 0.0, 0.5 },
    { 0.5, 0.0, -0.5, 0.0 },
    { 0.0, 0.5, 0.0, -0.5 }
};
static const double OUTPUT_BIAS[CHANNELS] = { 0.1, -0.2, 0.05, 0.3 };

typedef struct {
    double head_outputs[HEADS][BATCH_SIZE][TIME_STEPS][HEAD_SIZE];
    double attention_weights[HEADS][BATCH_SIZE][TIME_STEPS][TIME_STEPS];
    double concatenated[BATCH_SIZE][TIME_STEPS][CHANNELS];
    double output[BATCH_SIZE][TIME_STEPS][CHANNELS];
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
    for (size_t head = 0; head < HEADS; head++) {
        double queries[BATCH_SIZE][TIME_STEPS][HEAD_SIZE];
        double keys[BATCH_SIZE][TIME_STEPS][HEAD_SIZE];
        double values[BATCH_SIZE][TIME_STEPS][HEAD_SIZE];
        double scores[BATCH_SIZE][TIME_STEPS][TIME_STEPS];

        project(input, QUERY_WEIGHT[head], queries);
        project(input, KEY_WEIGHT[head], keys);
        project(input, VALUE_WEIGHT[head], values);

        for (size_t batch = 0; batch < BATCH_SIZE; batch++) {
            for (size_t time = 0; time < TIME_STEPS; time++) {
                for (size_t key_time = 0; key_time < TIME_STEPS; key_time++) {
                    double score = 0.0;
                    for (size_t channel = 0; channel < HEAD_SIZE; channel++) {
                        score += queries[batch][time][channel] * keys[batch][key_time][channel];
                    }
                    scores[batch][time][key_time] = key_time <= time
                        ? score / sqrt((double)HEAD_SIZE)
                        : -INFINITY;
                }
            }
        }

        for (size_t batch = 0; batch < BATCH_SIZE; batch++) {
            for (size_t time = 0; time < TIME_STEPS; time++) {
                double maximum = scores[batch][time][0];
                for (size_t key_time = 1; key_time <= time; key_time++) {
                    if (scores[batch][time][key_time] > maximum) {
                        maximum = scores[batch][time][key_time];
                    }
                }

                double total = 0.0;
                for (size_t key_time = 0; key_time < TIME_STEPS; key_time++) {
                    double score = scores[batch][time][key_time];
                    result->attention_weights[head][batch][time][key_time] =
                        isinf(score) ? 0.0 : exp(score - maximum);
                    total += result->attention_weights[head][batch][time][key_time];
                }
                for (size_t key_time = 0; key_time < TIME_STEPS; key_time++) {
                    result->attention_weights[head][batch][time][key_time] /= total;
                }
            }
        }

        for (size_t batch = 0; batch < BATCH_SIZE; batch++) {
            for (size_t time = 0; time < TIME_STEPS; time++) {
                for (size_t channel = 0; channel < HEAD_SIZE; channel++) {
                    result->head_outputs[head][batch][time][channel] = 0.0;
                    for (size_t key_time = 0; key_time < TIME_STEPS; key_time++) {
                        result->head_outputs[head][batch][time][channel] +=
                            result->attention_weights[head][batch][time][key_time]
                            * values[batch][key_time][channel];
                    }
                }
            }
        }
    }

    for (size_t batch = 0; batch < BATCH_SIZE; batch++) {
        for (size_t time = 0; time < TIME_STEPS; time++) {
            for (size_t head = 0; head < HEADS; head++) {
                for (size_t channel = 0; channel < HEAD_SIZE; channel++) {
                    result->concatenated[batch][time][head * HEAD_SIZE + channel] =
                        result->head_outputs[head][batch][time][channel];
                }
            }
        }
    }

    for (size_t batch = 0; batch < BATCH_SIZE; batch++) {
        for (size_t time = 0; time < TIME_STEPS; time++) {
            for (size_t output_channel = 0; output_channel < CHANNELS; output_channel++) {
                result->output[batch][time][output_channel] = OUTPUT_BIAS[output_channel];
                for (size_t input_channel = 0; input_channel < CHANNELS; input_channel++) {
                    result->output[batch][time][output_channel] +=
                        result->concatenated[batch][time][input_channel]
                        * OUTPUT_WEIGHT[output_channel][input_channel];
                }
            }
        }
    }
}

static void print_head_vectors(const char *label,
                               const double values[HEADS][BATCH_SIZE][TIME_STEPS][HEAD_SIZE]) {
    printf("%s, shape [heads, batch, time, head_size] = [%d, %d, %d, %d]:\n",
           label, HEADS, BATCH_SIZE, TIME_STEPS, HEAD_SIZE);
    for (size_t head = 0; head < HEADS; head++) {
        printf("  head %zu, batch 0:\n", head);
        for (size_t time = 0; time < TIME_STEPS; time++) {
            printf("    t=%zu [%.4f, %.4f]\n", time,
                   values[head][0][time][0], values[head][0][time][1]);
        }
    }
}

static void print_weights(
    const double values[HEADS][BATCH_SIZE][TIME_STEPS][TIME_STEPS]) {
    printf("Attention weights, shape [heads, batch, time, time] = [%d, %d, %d, %d] (batch 0):\n",
           HEADS, BATCH_SIZE, TIME_STEPS, TIME_STEPS);
    for (size_t head = 0; head < HEADS; head++) {
        printf("  head %zu:\n", head);
        for (size_t time = 0; time < TIME_STEPS; time++) {
            putchar(' ');
            for (size_t key_time = 0; key_time < TIME_STEPS; key_time++) {
                printf(" %8.4f", values[head][0][time][key_time]);
            }
            putchar('\n');
        }
    }
}

static void print_channels(const char *label,
                           const double values[BATCH_SIZE][TIME_STEPS][CHANNELS]) {
    printf("%s, shape [batch, time, channels] = [%d, %d, %d]:\n",
           label, BATCH_SIZE, TIME_STEPS, CHANNELS);
    for (size_t batch = 0; batch < BATCH_SIZE; batch++) {
        printf("  batch %zu\n", batch);
        for (size_t time = 0; time < TIME_STEPS; time++) {
            printf("    t=%zu [", time);
            for (size_t channel = 0; channel < CHANNELS; channel++) {
                printf("%s%.4f", channel == 0 ? "" : ", ",
                       values[batch][time][channel]);
            }
            puts("]");
        }
    }
}

static double maximum_prefix_difference(
    const double left[BATCH_SIZE][TIME_STEPS][CHANNELS],
    const double right[BATCH_SIZE][TIME_STEPS][CHANNELS],
    size_t batch, size_t prefix_length) {
    double maximum = 0.0;
    for (size_t time = 0; time < prefix_length; time++) {
        for (size_t channel = 0; channel < CHANNELS; channel++) {
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
        fprintf(stderr, "Usage: gpt1_multi_head_attention [all|1|2|3|4|1-2|1-3|1-4|2-4|3-4]\n");
        return EXIT_FAILURE;
    }

    AttentionResult result;
    forward(INPUT, &result);

    if (first_step <= 1) {
        puts("Step 1 - Run two independent causal attention heads");
        puts("Each head has its own query, key, and value projections and returns two channels per token.");
        puts("Attention weights use scaled query-key scores, a causal mask, and row-wise softmax.");
        print_weights(result.attention_weights);
        print_head_vectors("Individual head outputs", result.head_outputs);
    }

    if (first_step <= 2 && last_step >= 2) {
        puts("Step 2 - Concatenate the head outputs along the channel dimension");
        puts("Concatenation places head 0's two channels before head 1's two channels.");
        print_channels("Concatenated heads", result.concatenated);
        puts("The shape is [batch, time, heads * head_size] = [2, 4, 4].");
    }

    if (first_step <= 3 && last_step >= 3) {
        puts("Step 3 - Project the concatenated channels back to model width");
        puts("A learned linear projection mixes information from all heads; this example uses fixed weights.");
        print_channels("Projected multi-head output", result.output);
        puts("The model width is 4, so the shape returns to [batch, time, channels] = [2, 4, 4].");
        puts("Dropout is inactive in evaluation mode; C and Java show this deterministic forward pass.");
    }

    if (last_step >= 4) {
        double changed_input[BATCH_SIZE][TIME_STEPS][CHANNELS];
        memcpy(changed_input, INPUT, sizeof(changed_input));
        changed_input[0][3][0] += 100.0;
        AttentionResult changed_result;
        forward(changed_input, &changed_result);
        double earlier_difference =
            maximum_prefix_difference(result.output, changed_result.output, 0, 3);

        puts("Step 4 - Verify the combined block remains causal");
        printf("Change a future input token by +100: max change at earlier outputs = %.12f\n",
               earlier_difference);
        if (earlier_difference > 1e-10) {
            fputs("Causality check failed.\n", stderr);
            return EXIT_FAILURE;
        }
        puts("Causality check passed: neither head can pass future information to the past.");
    }

    return EXIT_SUCCESS;
}
