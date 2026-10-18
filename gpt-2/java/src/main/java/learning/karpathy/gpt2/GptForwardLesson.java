package learning.karpathy.gpt2;

import java.util.Locale;

/** A complete tiny GPT forward pass with fixed, inspectable parameters. */
public final class GptForwardLesson {
    private static final int SEQUENCE_LENGTH = 3;
    private static final int EMBEDDING_SIZE = 2;
    private static final int MLP_WIDTH = 4 * EMBEDDING_SIZE;
    private static final int VOCAB_SIZE = 4;
    private static final double EPSILON = 1e-5;
    private static final double SQRT_TWO_OVER_PI = Math.sqrt(2.0 / Math.PI);

    // Java stores the one-example batch as a T x C matrix; shape is conceptually (1, T, C).
    private static final int[] TOKEN_IDS = {1, 2, 3};
    private static final double[][] TOKEN_EMBEDDING_WEIGHT = {
        {0.10, 0.20}, {0.30, -0.10}, {-0.20, 0.40}, {0.05, 0.25}
    };
    private static final double[][] POSITION_EMBEDDING_WEIGHT = {
        {0.01, 0.00}, {0.00, 0.02}, {0.03, -0.01}
    };
    private static final double[][] IDENTITY = {{1.0, 0.0}, {0.0, 1.0}};
    private static final double[][] MLP_INPUT_WEIGHTS = {
        {0.20, -0.10}, {-0.10, 0.25}, {0.30, 0.05}, {0.00, 0.20},
        {-0.20, 0.10}, {0.10, -0.15}, {0.15, 0.00}, {-0.05, 0.30}
    };
    private static final double[][] MLP_OUTPUT_WEIGHTS = {
        {0.10, -0.10, 0.20, 0.00, 0.10, -0.15, 0.05, 0.10},
        {-0.20, 0.15, 0.05, 0.20, 0.10, 0.05, -0.05, 0.20}
    };
    private static final double[] NORM_GAMMA = {1.0, 1.0};
    private static final double[] NORM_BETA = {0.0, 0.0};

    private GptForwardLesson() {}

    public static void main(String[] args) {
        String part = args.length == 1 ? args[0] : "all";
        if (args.length > 1 || !part.matches("all|1|2|3")) {
            System.err.println("Usage: GptForwardLesson [all|1|2|3]");
            System.exit(1);
        }

        ForwardTrace trace = buildForward();
        int prediction = validateAndChooseNextToken(trace);
        if (prediction < 0) {
            throw new IllegalStateException("Forward-pass checks failed.");
        }

        if (part.equals("1") || part.equals("all")) {
            System.out.println("Step 1: map token IDs to vectors and add position vectors");
            System.out.printf("Token IDs (1 x %d): [%d, %d, %d]%n", SEQUENCE_LENGTH,
                    TOKEN_IDS[0], TOKEN_IDS[1], TOKEN_IDS[2]);
            printMatrix("Initial residual stream", trace.input());
            System.out.println("The singleton batch axis is omitted in Java storage; these vectors have shape (1, T, C).\n");
        }
        if (part.equals("2") || part.equals("all")) {
            System.out.println("Step 2: run one decoder block and preserve the sequence shape");
            printHidden("MLP hidden state", trace.mlpHidden());
            printMatrix("Decoder block output", trace.blockOutput());
            System.out.println("Attention mixes positions; the MLP transforms each position independently.\n");
        }
        if (part.equals("3") || part.equals("all")) {
            System.out.println("Step 3: final LayerNorm, tied output projection, and next-token choice");
            printMatrix("Final normalized hidden states", trace.finalNormalized());
            printLogits(trace.logits());
            System.out.println("Toy next-token choice at the last position: token ID " + prediction);
            System.out.println("Logits are unnormalized scores. These fixed toy weights are not a trained language model.\n");
        }
    }

    private static ForwardTrace buildForward() {
        double[][] tokenVectors = new double[SEQUENCE_LENGTH][EMBEDDING_SIZE];
        double[][] input = new double[SEQUENCE_LENGTH][EMBEDDING_SIZE];
        for (int position = 0; position < SEQUENCE_LENGTH; position++) {
            int tokenId = TOKEN_IDS[position];
            for (int channel = 0; channel < EMBEDDING_SIZE; channel++) {
                tokenVectors[position][channel] = TOKEN_EMBEDDING_WEIGHT[tokenId][channel];
                input[position][channel] = tokenVectors[position][channel]
                        + POSITION_EMBEDDING_WEIGHT[position][channel];
            }
        }

        double[][] normalizedInput = layerNorm(input);
        double[][] query = project(normalizedInput, IDENTITY);
        double[][] key = project(normalizedInput, IDENTITY);
        double[][] value = project(normalizedInput, IDENTITY);
        AttentionTrace attention = causalAttention(query, key, value);
        double[][] afterAttention = add(input, attention.context());

        double[][] normalizedForMlp = layerNorm(afterAttention);
        double[][] mlpPreActivation = linear(normalizedForMlp, MLP_INPUT_WEIGHTS);
        double[][] mlpHidden = geluTanh(mlpPreActivation);
        double[][] mlpBranch = linear(mlpHidden, MLP_OUTPUT_WEIGHTS);
        double[][] blockOutput = add(afterAttention, mlpBranch);

        double[][] finalNormalized = layerNorm(blockOutput);
        // Weight tying: use the embedding table rows as output projection weights.
        double[][] logits = linear(finalNormalized, TOKEN_EMBEDDING_WEIGHT);
        return new ForwardTrace(tokenVectors, input, attention.weights(), mlpHidden,
                blockOutput, finalNormalized, logits);
    }

    private static double[][] layerNorm(double[][] values) {
        double[][] normalized = new double[SEQUENCE_LENGTH][EMBEDDING_SIZE];
        for (int position = 0; position < SEQUENCE_LENGTH; position++) {
            double mean = 0.0;
            for (int channel = 0; channel < EMBEDDING_SIZE; channel++) {
                mean += values[position][channel];
            }
            mean /= EMBEDDING_SIZE;

            double variance = 0.0;
            for (int channel = 0; channel < EMBEDDING_SIZE; channel++) {
                double difference = values[position][channel] - mean;
                variance += difference * difference;
            }
            variance /= EMBEDDING_SIZE;
            double inverseStddev = 1.0 / Math.sqrt(variance + EPSILON);
            for (int channel = 0; channel < EMBEDDING_SIZE; channel++) {
                normalized[position][channel] = (values[position][channel] - mean)
                        * inverseStddev * NORM_GAMMA[channel] + NORM_BETA[channel];
            }
        }
        return normalized;
    }

    /** Linear transformation with weights stored as (output channel, input channel). */
    private static double[][] linear(double[][] values, double[][] weights) {
        double[][] projected = new double[SEQUENCE_LENGTH][weights.length];
        for (int position = 0; position < SEQUENCE_LENGTH; position++) {
            for (int outputChannel = 0; outputChannel < weights.length; outputChannel++) {
                for (int inputChannel = 0; inputChannel < values[position].length; inputChannel++) {
                    projected[position][outputChannel] += values[position][inputChannel]
                            * weights[outputChannel][inputChannel];
                }
            }
        }
        return projected;
    }

    private static double[][] project(double[][] values, double[][] weights) {
        return linear(values, weights);
    }

    private static AttentionTrace causalAttention(double[][] query, double[][] key, double[][] value) {
        double[][] weights = new double[SEQUENCE_LENGTH][SEQUENCE_LENGTH];
        double[][] context = new double[SEQUENCE_LENGTH][EMBEDDING_SIZE];
        double scale = Math.sqrt(EMBEDDING_SIZE);
        for (int queryPosition = 0; queryPosition < SEQUENCE_LENGTH; queryPosition++) {
            double[] logits = new double[SEQUENCE_LENGTH];
            double maximum = Double.NEGATIVE_INFINITY;
            for (int keyPosition = 0; keyPosition < SEQUENCE_LENGTH; keyPosition++) {
                if (keyPosition > queryPosition) {
                    logits[keyPosition] = Double.NEGATIVE_INFINITY;
                } else {
                    for (int channel = 0; channel < EMBEDDING_SIZE; channel++) {
                        logits[keyPosition] += query[queryPosition][channel] * key[keyPosition][channel];
                    }
                    logits[keyPosition] /= scale;
                }
                maximum = Math.max(maximum, logits[keyPosition]);
            }

            double denominator = 0.0;
            for (int keyPosition = 0; keyPosition < SEQUENCE_LENGTH; keyPosition++) {
                weights[queryPosition][keyPosition] = Math.exp(logits[keyPosition] - maximum);
                denominator += weights[queryPosition][keyPosition];
            }
            for (int keyPosition = 0; keyPosition < SEQUENCE_LENGTH; keyPosition++) {
                weights[queryPosition][keyPosition] /= denominator;
                for (int channel = 0; channel < EMBEDDING_SIZE; channel++) {
                    context[queryPosition][channel] += weights[queryPosition][keyPosition]
                            * value[keyPosition][channel];
                }
            }
        }
        return new AttentionTrace(weights, context);
    }

    private static double[][] geluTanh(double[][] values) {
        double[][] activated = new double[SEQUENCE_LENGTH][MLP_WIDTH];
        for (int position = 0; position < SEQUENCE_LENGTH; position++) {
            for (int neuron = 0; neuron < MLP_WIDTH; neuron++) {
                double value = values[position][neuron];
                double cubicTerm = 0.044715 * value * value * value;
                activated[position][neuron] = 0.5 * value
                        * (1.0 + Math.tanh(SQRT_TWO_OVER_PI * (value + cubicTerm)));
            }
        }
        return activated;
    }

    private static double[][] add(double[][] left, double[][] right) {
        double[][] sum = new double[SEQUENCE_LENGTH][EMBEDDING_SIZE];
        for (int position = 0; position < SEQUENCE_LENGTH; position++) {
            for (int channel = 0; channel < EMBEDDING_SIZE; channel++) {
                sum[position][channel] = left[position][channel] + right[position][channel];
            }
        }
        return sum;
    }

    private static int validateAndChooseNextToken(ForwardTrace trace) {
        for (int position = 0; position < SEQUENCE_LENGTH; position++) {
            double rowSum = 0.0;
            for (int keyPosition = 0; keyPosition < SEQUENCE_LENGTH; keyPosition++) {
                double weight = trace.attentionWeights()[position][keyPosition];
                if (!Double.isFinite(weight) || (keyPosition > position && Math.abs(weight) > 1e-12)) {
                    return -1;
                }
                rowSum += weight;
            }
            if (Math.abs(rowSum - 1.0) > 1e-12) return -1;
            for (int token = 0; token < VOCAB_SIZE; token++) {
                if (!Double.isFinite(trace.logits()[position][token])) return -1;
            }
        }

        int prediction = 0;
        for (int token = 1; token < VOCAB_SIZE; token++) {
            if (trace.logits()[SEQUENCE_LENGTH - 1][token]
                    > trace.logits()[SEQUENCE_LENGTH - 1][prediction]) {
                prediction = token;
            }
        }
        return prediction;
    }

    private static void printMatrix(String label, double[][] values) {
        System.out.printf("%s (1 x %d x %d):%n", label, values.length, values[0].length);
        for (double[] row : values) printRow(row);
    }

    private static void printHidden(String label, double[][] values) {
        System.out.printf("%s (1 x %d x %d):%n", label, values.length, values[0].length);
        for (double[] row : values) printRow(row);
    }

    private static void printLogits(double[][] values) {
        System.out.printf("Vocabulary logits (1 x %d x %d):%n", values.length, values[0].length);
        for (double[] row : values) printRow(row);
    }

    private static void printRow(double[] values) {
        System.out.print("  [");
        for (int index = 0; index < values.length; index++) {
            if (index > 0) System.out.print(", ");
            System.out.print(String.format(Locale.ROOT, "%.4f", values[index]));
        }
        System.out.println("]");
    }

    private record AttentionTrace(double[][] weights, double[][] context) {}

    private record ForwardTrace(
            double[][] tokenVectors,
            double[][] input,
            double[][] attentionWeights,
            double[][] mlpHidden,
            double[][] blockOutput,
            double[][] finalNormalized,
            double[][] logits) {}
}
