#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { TIME_STEPS = 4, CHANNELS = 2 };

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

static void print_vectors(const char *label, const double vectors[TIME_STEPS][CHANNELS]) {
    printf("%s, shape [%d, %d]:\n", label, TIME_STEPS, CHANNELS);
    for (size_t time = 0; time < TIME_STEPS; time++) {
        printf("  t=%zu [%.3f, %.3f]\n", time, vectors[time][0], vectors[time][1]);
    }
}

static void print_matrix(const char *label, const double matrix[TIME_STEPS][TIME_STEPS]) {
    printf("%s, shape [%d, %d]:\n", label, TIME_STEPS, TIME_STEPS);
    for (size_t row = 0; row < TIME_STEPS; row++) {
        putchar(' ');
        for (size_t column = 0; column < TIME_STEPS; column++) {
            printf(" %8.4f", matrix[row][column]);
        }
        putchar('\n');
    }
}

static void average_visible_past(const double input[TIME_STEPS][CHANNELS],
                                 double output[TIME_STEPS][CHANNELS]) {
    for (size_t time = 0; time < TIME_STEPS; time++) {
        for (size_t channel = 0; channel < CHANNELS; channel++) {
            double total = 0.0;
            for (size_t past = 0; past <= time; past++) {
                total += input[past][channel];
            }
            output[time][channel] = total / (double)(time + 1);
        }
    }
}

static void make_causal_average_weights(double weights[TIME_STEPS][TIME_STEPS]) {
    for (size_t time = 0; time < TIME_STEPS; time++) {
        for (size_t past = 0; past < TIME_STEPS; past++) {
            weights[time][past] = past <= time ? 1.0 / (double)(time + 1) : 0.0;
        }
    }
}

static void apply_weights(const double weights[TIME_STEPS][TIME_STEPS],
                          const double values[TIME_STEPS][CHANNELS],
                          double output[TIME_STEPS][CHANNELS]) {
    for (size_t time = 0; time < TIME_STEPS; time++) {
        for (size_t channel = 0; channel < CHANNELS; channel++) {
            output[time][channel] = 0.0;
            for (size_t past = 0; past < TIME_STEPS; past++) {
                output[time][channel] += weights[time][past] * values[past][channel];
            }
        }
    }
}

static void softmax_rows(const double scores[TIME_STEPS][TIME_STEPS],
                         double weights[TIME_STEPS][TIME_STEPS]) {
    for (size_t time = 0; time < TIME_STEPS; time++) {
        double maximum = scores[time][0];
        for (size_t key = 1; key < TIME_STEPS; key++) {
            if (scores[time][key] > maximum) maximum = scores[time][key];
        }

        double total = 0.0;
        for (size_t key = 0; key < TIME_STEPS; key++) {
            weights[time][key] = exp(scores[time][key] - maximum);
            total += weights[time][key];
        }
        for (size_t key = 0; key < TIME_STEPS; key++) {
            weights[time][key] /= total;
        }
    }
}

static double maximum_difference(const double left[TIME_STEPS][CHANNELS],
                                 const double right[TIME_STEPS][CHANNELS]) {
    double maximum = 0.0;
    for (size_t time = 0; time < TIME_STEPS; time++) {
        for (size_t channel = 0; channel < CHANNELS; channel++) {
            double difference = fabs(left[time][channel] - right[time][channel]);
            if (difference > maximum) maximum = difference;
        }
    }
    return maximum;
}

static void project(const double input[TIME_STEPS][CHANNELS],
                    const double matrix[CHANNELS][CHANNELS],
                    double output[TIME_STEPS][CHANNELS]) {
    for (size_t time = 0; time < TIME_STEPS; time++) {
        for (size_t output_channel = 0; output_channel < CHANNELS; output_channel++) {
            output[time][output_channel] = 0.0;
            for (size_t input_channel = 0; input_channel < CHANNELS; input_channel++) {
                output[time][output_channel] +=
                    input[time][input_channel] * matrix[input_channel][output_channel];
            }
        }
    }
}

int main(int argc, char *argv[]) {
    const char *part = argc == 1 ? "all" : argv[1];
    int first_step;
    int last_step;
    if (argc > 2 || !parse_part(part, &first_step, &last_step)) {
        fprintf(stderr, "Usage: gpt1_attention_math [all|1|2|3|4|1-2|1-3|1-4|2-4|3-4]\n");
        return EXIT_FAILURE;
    }

    /* Four short token vectors keep every intermediate small enough to inspect. */
    const double input[TIME_STEPS][CHANNELS] = {
        {1.0, 0.0}, {0.0, 2.0}, {3.0, 1.0}, {2.0, 4.0}
    };
    double loop_output[TIME_STEPS][CHANNELS];
    double causal_weights[TIME_STEPS][TIME_STEPS];
    double matrix_output[TIME_STEPS][CHANNELS];
    double zero_scores[TIME_STEPS][TIME_STEPS] = {{0.0}};
    double uniform_weights[TIME_STEPS][TIME_STEPS];
    double softmax_output[TIME_STEPS][CHANNELS];

    average_visible_past(input, loop_output);
    make_causal_average_weights(causal_weights);
    apply_weights(causal_weights, input, matrix_output);

    /* Future positions receive negative infinity, so softmax gives them zero weight. */
    for (size_t time = 0; time < TIME_STEPS; time++) {
        for (size_t key = time + 1; key < TIME_STEPS; key++) {
            zero_scores[time][key] = -INFINITY;
        }
    }
    softmax_rows(zero_scores, uniform_weights);
    apply_weights(uniform_weights, input, softmax_output);

    if (first_step <= 1) {
        puts("Step 1 - Average each token with the visible past using loops");
        print_vectors("Input vectors", input);
        print_vectors("Causal averages", loop_output);
        puts("At time t, this loop reads positions 0 through t and cannot see the future.");
    }

    if (first_step <= 2 && last_step >= 2) {
        puts("Step 2 - Write the same operation as a matrix multiplication");
        print_matrix("Lower-triangular averaging weights", causal_weights);
        print_vectors("Weights x input", matrix_output);
        printf("Maximum difference from the loop version: %.12f\n",
               maximum_difference(loop_output, matrix_output));
    }

    if (first_step <= 3 && last_step >= 3) {
        puts("Step 3 - Use a causal mask and softmax to create the weights");
        print_matrix("Scores after masking future positions", zero_scores);
        print_matrix("Softmax weights", uniform_weights);
        printf("Maximum difference from the loop version: %.12f\n",
               maximum_difference(loop_output, softmax_output));
    }

    if (last_step >= 4) {
        const double query_projection[CHANNELS][CHANNELS] = {
            {1.0, 0.5}, {-0.5, 1.0}
        };
        const double key_projection[CHANNELS][CHANNELS] = {
            {0.5, -0.5}, {1.0, 0.5}
        };
        double queries[TIME_STEPS][CHANNELS];
        double keys[TIME_STEPS][CHANNELS];
        double attention_scores[TIME_STEPS][TIME_STEPS];
        double attention_weights[TIME_STEPS][TIME_STEPS];
        double attention_output[TIME_STEPS][CHANNELS];

        project(input, query_projection, queries);
        project(input, key_projection, keys);
        for (size_t time = 0; time < TIME_STEPS; time++) {
            for (size_t key = 0; key < TIME_STEPS; key++) {
                double score = 0.0;
                for (size_t channel = 0; channel < CHANNELS; channel++) {
                    score += queries[time][channel] * keys[key][channel];
                }
                attention_scores[time][key] = key <= time
                    ? score / sqrt((double)CHANNELS)
                    : -INFINITY;
            }
        }
        softmax_rows(attention_scores, attention_weights);
        apply_weights(attention_weights, input, attention_output);

        puts("Step 4 - Let query-key similarity choose the causal weights");
        puts("The fixed projection matrices stand in for learned query/key layers; input vectors act as values.");
        print_vectors("Queries", queries);
        print_vectors("Keys", keys);
        print_matrix("Scaled query-key scores after the causal mask", attention_scores);
        print_matrix("Input-dependent attention weights", attention_weights);
        print_vectors("Weighted value vectors", attention_output);
        puts("Each row attends only to its own position and earlier positions; later positions have weight zero.");
    }

    return EXIT_SUCCESS;
}
