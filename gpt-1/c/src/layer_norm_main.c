#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { BATCH_SIZE = 2, TIME_STEPS = 2, CHANNELS = 4 };

static const double INPUT[BATCH_SIZE][TIME_STEPS][CHANNELS] = {
    { { 1.0, 2.0, 3.0, 4.0 }, { 2.0, 4.0, 6.0, 8.0 } },
    { { 10.0, 11.0, 12.0, 13.0 }, { 3.0, 1.0, 4.0, 2.0 } }
};
static const double GAMMA[CHANNELS] = { 1.0, 0.5, 2.0, -1.0 };
static const double BETA[CHANNELS] = { 0.0, 1.0, -1.0, 0.5 };
static const double EPSILON = 1e-5;

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

static void row_statistics(const double row[CHANNELS], double *mean, double *variance) {
    *mean = 0.0;
    for (size_t channel = 0; channel < CHANNELS; channel++) {
        *mean += row[channel];
    }
    *mean /= (double)CHANNELS;

    *variance = 0.0;
    for (size_t channel = 0; channel < CHANNELS; channel++) {
        double centered = row[channel] - *mean;
        *variance += centered * centered;
    }
    *variance /= (double)CHANNELS; /* Population variance, as in nn.LayerNorm. */
}

static void normalize_input(double normalized[BATCH_SIZE][TIME_STEPS][CHANNELS]) {
    for (size_t batch = 0; batch < BATCH_SIZE; batch++) {
        for (size_t time = 0; time < TIME_STEPS; time++) {
            double mean;
            double variance;
            row_statistics(INPUT[batch][time], &mean, &variance);
            for (size_t channel = 0; channel < CHANNELS; channel++) {
                normalized[batch][time][channel] =
                    (INPUT[batch][time][channel] - mean) / sqrt(variance + EPSILON);
            }
        }
    }
}

static void print_vector(const double values[CHANNELS]) {
    printf("[%.4f, %.4f, %.4f, %.4f]", values[0], values[1], values[2], values[3]);
}

static void print_token_vectors(const char *label,
                                const double values[BATCH_SIZE][TIME_STEPS][CHANNELS]) {
    puts(label);
    for (size_t batch = 0; batch < BATCH_SIZE; batch++) {
        for (size_t time = 0; time < TIME_STEPS; time++) {
            printf("  batch %zu, time %zu: ", batch, time);
            print_vector(values[batch][time]);
            putchar('\n');
        }
    }
}

int main(int argc, char *argv[]) {
    const char *part = argc == 1 ? "all" : argv[1];
    int first_step;
    int last_step;
    if (argc > 2 || !parse_part(part, &first_step, &last_step)) {
        fprintf(stderr, "Usage: gpt1_layer_norm [all|1|2|3|1-2|1-3]\n");
        return EXIT_FAILURE;
    }

    double normalized[BATCH_SIZE][TIME_STEPS][CHANNELS];
    normalize_input(normalized);

    if (first_step <= 1) {
        puts("Step 1 - Choose the dimension being normalized");
        puts("Input shape: (batch=2, time=2, channels=4)");
        puts("LayerNorm computes statistics over each token's channels independently.");
        for (size_t batch = 0; batch < BATCH_SIZE; batch++) {
            for (size_t time = 0; time < TIME_STEPS; time++) {
                double mean;
                double variance;
                row_statistics(INPUT[batch][time], &mean, &variance);
                printf("  batch %zu, time %zu input ", batch, time);
                print_vector(INPUT[batch][time]);
                printf(" -> mean %.4f, population variance %.4f\n", mean, variance);
            }
        }
        puts("Other batch examples and time positions do not affect these statistics.");
    }

    if (first_step <= 2 && last_step >= 2) {
        puts("Step 2 - Center and rescale each token vector");
        puts("x_hat = (x - mean) / sqrt(population_variance + eps), with eps = 1e-5");
        print_token_vectors("Normalized token vectors:", normalized);
        for (size_t batch = 0; batch < BATCH_SIZE; batch++) {
            for (size_t time = 0; time < TIME_STEPS; time++) {
                double mean;
                double variance;
                row_statistics(normalized[batch][time], &mean, &variance);
                printf("  normalized mean %.4f, population std %.4f\n",
                       mean, sqrt(variance));
            }
        }
        double channel_zero_sum = 0.0;
        for (size_t batch = 0; batch < BATCH_SIZE; batch++) {
            for (size_t time = 0; time < TIME_STEPS; time++) {
                channel_zero_sum += normalized[batch][time][0];
            }
        }
        printf("Mean of channel 0 across all tokens (not constrained): %.4f\n",
               channel_zero_sum / (BATCH_SIZE * TIME_STEPS));
    }

    if (last_step >= 3) {
        double output[BATCH_SIZE][TIME_STEPS][CHANNELS];
        for (size_t batch = 0; batch < BATCH_SIZE; batch++) {
            for (size_t time = 0; time < TIME_STEPS; time++) {
                for (size_t channel = 0; channel < CHANNELS; channel++) {
                    output[batch][time][channel] =
                        GAMMA[channel] * normalized[batch][time][channel] + BETA[channel];
                }
            }
        }

        puts("Step 3 - Apply a learned scale and shift per channel");
        printf("gamma: ");
        print_vector(GAMMA);
        printf("\nbeta:  ");
        print_vector(BETA);
        putchar('\n');
        puts("LayerNorm output: gamma * x_hat + beta");
        print_token_vectors("Output token vectors:", output);
        puts("These example gamma/beta values are fixed here; training would update them by gradient descent.");
    }

    return EXIT_SUCCESS;
}
