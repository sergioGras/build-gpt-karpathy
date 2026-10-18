#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { BATCH_SIZE = 2, TIME_STEPS = 4, CHANNELS = 4, HIDDEN_SIZE = 4 * CHANNELS };
static const double EPSILON = 1e-5;

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

/* Fixed output of Lesson 10's multi-head attention on LayerNorm(INPUT). */
static const double ATTENTION_OUTPUT[BATCH_SIZE][TIME_STEPS][CHANNELS] = {
    {
        { 0.9944236133126180, -0.0881970483359226, 0.0500000000000000, -1.6006501782893132 },
        { 0.3933738827285683, 0.1954140878650754, 0.2038379239277406, -0.6362878460652932 },
        { -0.5175716661695421, -1.3505289024072524, 1.4923256127023328, -0.1986207426936189 },
        { 0.6142816558095826, -0.5870337399944243, -0.4857832613323823, 1.4064524206079174 }
    },
    {
        { 0.3236059033281545, -0.5354088549922318, -0.6208177099844634, 1.0826206616485405 },
        { 0.6468403211190350, 0.2777975158844621, 0.5677887061817602, -1.9983469601638093 },
        { 0.6771942604404187, -0.9952272543893472, 0.1465132161629877, -0.0665467975441047 },
        { -0.2972420175988929, 0.2774110104487157, 0.7685343078948748, -0.3120942225247459 }
    }
};

static double first_weight[HIDDEN_SIZE][CHANNELS];
static double first_bias[HIDDEN_SIZE];
static double second_weight[CHANNELS][HIDDEN_SIZE];
static const double second_bias[CHANNELS] = { 0.02, -0.01, 0.03, -0.02 };

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

static void layer_norm(const double input[BATCH_SIZE][TIME_STEPS][CHANNELS],
                       double output[BATCH_SIZE][TIME_STEPS][CHANNELS]) {
    for (size_t batch = 0; batch < BATCH_SIZE; batch++) {
        for (size_t time = 0; time < TIME_STEPS; time++) {
            double mean = 0.0;
            for (size_t channel = 0; channel < CHANNELS; channel++) {
                mean += input[batch][time][channel];
            }
            mean /= (double)CHANNELS;

            double variance = 0.0;
            for (size_t channel = 0; channel < CHANNELS; channel++) {
                double centered = input[batch][time][channel] - mean;
                variance += centered * centered;
            }
            variance /= (double)CHANNELS;

            for (size_t channel = 0; channel < CHANNELS; channel++) {
                double normalized = (input[batch][time][channel] - mean)
                                  / sqrt(variance + EPSILON);
                output[batch][time][channel] = normalized; /* gamma=1, beta=0 */
            }
        }
    }
}

static void initialize_feed_forward_weights(void) {
    for (size_t hidden = 0; hidden < HIDDEN_SIZE; hidden++) {
        first_bias[hidden] = ((int)(hidden % 3) - 1) * 0.05;
        for (size_t channel = 0; channel < CHANNELS; channel++) {
            int pattern = (int)(((hidden + 1) * (channel + 2)) % 7) - 3;
            first_weight[hidden][channel] = pattern * 0.1;
        }
    }
    for (size_t channel = 0; channel < CHANNELS; channel++) {
        for (size_t hidden = 0; hidden < HIDDEN_SIZE; hidden++) {
            int pattern = (int)(((channel + 2) * (hidden + 1)) % 5) - 2;
            second_weight[channel][hidden] = pattern * 0.05;
        }
    }
}

static void feed_forward(
    const double input[BATCH_SIZE][TIME_STEPS][CHANNELS],
    double hidden_output[BATCH_SIZE][TIME_STEPS][HIDDEN_SIZE],
    double output[BATCH_SIZE][TIME_STEPS][CHANNELS]) {
    for (size_t batch = 0; batch < BATCH_SIZE; batch++) {
        for (size_t time = 0; time < TIME_STEPS; time++) {
            for (size_t hidden = 0; hidden < HIDDEN_SIZE; hidden++) {
                double value = first_bias[hidden];
                for (size_t channel = 0; channel < CHANNELS; channel++) {
                    value += input[batch][time][channel] * first_weight[hidden][channel];
                }
                hidden_output[batch][time][hidden] = value > 0.0 ? value : 0.0; /* ReLU */
            }

            for (size_t channel = 0; channel < CHANNELS; channel++) {
                double value = second_bias[channel];
                for (size_t hidden = 0; hidden < HIDDEN_SIZE; hidden++) {
                    value += hidden_output[batch][time][hidden]
                           * second_weight[channel][hidden];
                }
                output[batch][time][channel] = value; /* Dropout is inactive in evaluation mode. */
            }
        }
    }
}

static void add_residual(const double left[BATCH_SIZE][TIME_STEPS][CHANNELS],
                         const double right[BATCH_SIZE][TIME_STEPS][CHANNELS],
                         double output[BATCH_SIZE][TIME_STEPS][CHANNELS]) {
    for (size_t batch = 0; batch < BATCH_SIZE; batch++) {
        for (size_t time = 0; time < TIME_STEPS; time++) {
            for (size_t channel = 0; channel < CHANNELS; channel++) {
                output[batch][time][channel] =
                    left[batch][time][channel] + right[batch][time][channel];
            }
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
                printf("%s%.4f", channel == 0 ? "" : ", ", values[batch][time][channel]);
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
        fprintf(stderr, "Usage: gpt1_transformer_block [all|1|2|3|4|1-2|1-3|1-4|2-4|3-4]\n");
        return EXIT_FAILURE;
    }

    double normalized_input[BATCH_SIZE][TIME_STEPS][CHANNELS];
    double after_attention[BATCH_SIZE][TIME_STEPS][CHANNELS];
    double normalized_after_attention[BATCH_SIZE][TIME_STEPS][CHANNELS];
    double hidden[BATCH_SIZE][TIME_STEPS][HIDDEN_SIZE];
    double feed_forward_output[BATCH_SIZE][TIME_STEPS][CHANNELS];
    double block_output[BATCH_SIZE][TIME_STEPS][CHANNELS];

    layer_norm(INPUT, normalized_input);
    add_residual(INPUT, ATTENTION_OUTPUT, after_attention);
    layer_norm(after_attention, normalized_after_attention);
    initialize_feed_forward_weights();
    feed_forward(normalized_after_attention, hidden, feed_forward_output);
    add_residual(after_attention, feed_forward_output, block_output);

    if (first_step <= 1) {
        puts("Step 1 - Normalize each token before its sublayer");
        puts("LayerNorm uses each token's channel mean and population variance; gamma=1 and beta=0 here.");
        print_channels("LayerNorm(input), supplied to Lesson 10 attention", normalized_input);
        puts("The attention result below is the fixed Lesson 10 example output for this normalized input.");
    }

    if (first_step <= 2 && last_step >= 2) {
        puts("Step 2 - Add the attention result through a residual connection");
        print_channels("Attention branch output", ATTENTION_OUTPUT);
        print_channels("After attention residual: input + attention output", after_attention);
        puts("The residual keeps the original representation available to later layers.");
    }

    if (first_step <= 3 && last_step >= 3) {
        puts("Step 3 - Normalize, expand, apply ReLU, and project back");
        puts("The position-wise feed-forward network expands 4 channels to 16, then contracts to 4.");
        print_channels("LayerNorm after the attention residual", normalized_after_attention);
        printf("Expanded ReLU representation shape = [%d, %d, %d]. First token: [",
               BATCH_SIZE, TIME_STEPS, HIDDEN_SIZE);
        for (size_t hidden_channel = 0; hidden_channel < HIDDEN_SIZE; hidden_channel++) {
            printf("%s%.4f", hidden_channel == 0 ? "" : ", ", hidden[0][0][hidden_channel]);
        }
        puts("]");
        print_channels("Feed-forward output", feed_forward_output);
        puts("Dropout follows the second linear layer in training; it is inactive for this evaluation example.");
    }

    if (last_step >= 4) {
        double changed_input[BATCH_SIZE][TIME_STEPS][CHANNELS];
        double changed_hidden[BATCH_SIZE][TIME_STEPS][HIDDEN_SIZE];
        double changed_feed_forward[BATCH_SIZE][TIME_STEPS][CHANNELS];
        memcpy(changed_input, normalized_after_attention, sizeof(changed_input));
        changed_input[0][3][0] += 10.0;
        feed_forward(changed_input, changed_hidden, changed_feed_forward);
        double earlier_difference = maximum_prefix_difference(
            feed_forward_output, changed_feed_forward, 0, 3);

        puts("Step 4 - Add the feed-forward residual and inspect the block output");
        print_channels("Transformer block output", block_output);
        printf("Block output shape is unchanged: [batch, time, channels] = [%d, %d, %d].\n",
               BATCH_SIZE, TIME_STEPS, CHANNELS);
        printf("Change one token entering the feed-forward network: max change at other tokens = %.12f\n",
               earlier_difference);
        if (earlier_difference > 1e-10) {
            fputs("Position-wise feed-forward check failed.\n", stderr);
            return EXIT_FAILURE;
        }
        puts("Check passed: the feed-forward network mixes channels within each token, not information across time.");
    }

    return EXIT_SUCCESS;
}
