package learning.karpathy.gpt2;

import java.util.Locale;

/** A small GPT-2-style pre-normalized decoder block with explicit toy weights. */
public final class BlockLesson {
    private static final int SEQUENCE_LENGTH = 3;
    private static final int EMBEDDING_SIZE = 2;
    private static final int MLP_WIDTH = 4 * EMBEDDING_SIZE;
    private static final double LAYER_NORM_EPSILON = 1e-5;
    private static final double SQRT_TWO_OVER_PI = Math.sqrt(2.0 / Math.PI);

    // Reuse the first residual-stream sequence from Lesson 4.
    private static final double[][] INPUT = {
        {1.1, 0.0}, {0.0, 1.2}, {1.3, 1.3}
    };

    // Identity projections keep this lesson focused on decoder-block structure.
    private static final double[][] QUERY_WEIGHTS = {{1.0, 0.0}, {0.0, 1.0}};
    private static final double[][] KEY_WEIGHTS = {{1.0, 0.0}, {0.0, 1.0}};
    private static final double[][] VALUE_WEIGHTS = {{1.0, 0.0}, {0.0, 1.0}};
    private static final double[][] ATTENTION_OUTPUT_WEIGHTS = {{1.0, 0.0}, {0.0, 1.0}};

    // LayerNorm starts with gamma=1 and beta=0; training can learn both.
    private static final double[] NORM1_GAMMA = {1.0, 1.0};
    private static final double[] NORM1_BETA = {0.0, 0.0};
    private static final double[] NORM2_GAMMA = {1.0, 1.0};
    private static final double[] NORM2_BETA = {0.0, 0.0};

    // Linear weights use (output channel, input channel) order, as in nn.Linear.
    private static final double[][] MLP_INPUT_WEIGHTS = {
        {0.20, -0.10}, {-0.10, 0.25}, {0.30, 0.05}, {0.00, 0.20},
        {-0.20, 0.10}, {0.10, -0.15}, {0.15, 0.00}, {-0.05, 0.30}
    };
    private static final double[] MLP_INPUT_BIAS = new double[MLP_WIDTH];
    private static final double[][] MLP_OUTPUT_WEIGHTS = {
        {0.10, -0.10, 0.20, 0.00, 0.10, -0.15, 0.05, 0.10},
        {-0.20, 0.15, 0.05, 0.20, 0.10, 0.05, -0.05, 0.20}
    };
    private static final double[] MLP_OUTPUT_BIAS = new double[EMBEDDING_SIZE];

    private BlockLesson() {}

    public static void main(String[] args) {
        String part = args.length == 1 ? args[0] : "all";
        if (args.length > 1 || !part.matches("all|1|2|3")) {
            System.err.println("Usage: BlockLesson [all|1|2|3]");
            System.exit(1);
        }

        BlockTrace trace = buildBlock(INPUT);
        if (part.equals("1") || part.equals("all")) {
            System.out.println("Step 1: normalize, attend causally, and add the attention residual");
            printEmbeddingMatrix("LayerNorm 1 output", trace.normalizedInput());
            printAttentionMatrix("Causal attention weights", trace.attentionWeights());
            printEmbeddingMatrix("Attention branch output", trace.attentionBranch());
            printEmbeddingMatrix("After x = x + attention(LayerNorm 1(x))", trace.afterAttention());
            System.out.println("The skip path preserves the incoming representation while attention adds context.\n");
        }
        if (part.equals("2") || part.equals("all")) {
            System.out.println("Step 2: normalize again, expand through the MLP, and add its residual");
            printEmbeddingMatrix("LayerNorm 2 output", trace.normalizedForMlp());
            printHiddenMatrix("MLP linear layer before GELU (T, 4C)", trace.mlpPreActivation());
            printHiddenMatrix("MLP after GELU", trace.mlpActivated());
            printEmbeddingMatrix("MLP branch output", trace.mlpBranch());
            System.out.println("The MLP changes each position independently; it does not mix information across tokens.\n");
        }
        if (part.equals("3") || part.equals("all")) {
            System.out.println("Step 3: inspect the complete decoder-block output");
            printEmbeddingMatrix("Block output after the second residual", trace.output());
            boolean valid = residualEquationsHold(INPUT, trace);
            System.out.println("Both residual equations hold and all outputs are finite: " + (valid ? "yes" : "no"));
            if (!valid) {
                throw new IllegalStateException("Decoder-block residual check failed.");
            }
            System.out.println("GPT-2 stacks 12 such blocks; the final normalization and vocabulary projection come afterward.\n");
        }
    }

    private static BlockTrace buildBlock(double[][] input) {
        double[][] normalizedInput = layerNorm(input, NORM1_GAMMA, NORM1_BETA);
        double[][] query = project(normalizedInput, QUERY_WEIGHTS);
        double[][] key = project(normalizedInput, KEY_WEIGHTS);
        double[][] value = project(normalizedInput, VALUE_WEIGHTS);
        AttentionTrace attention = causalAttention(query, key, value);
        double[][] attentionBranch = project(attention.context(), ATTENTION_OUTPUT_WEIGHTS);
        double[][] afterAttention = add(input, attentionBranch);

        double[][] normalizedForMlp = layerNorm(afterAttention, NORM2_GAMMA, NORM2_BETA);
        double[][] mlpPreActivation = mlpLinearIn(normalizedForMlp);
        double[][] mlpActivated = geluTanh(mlpPreActivation);
        double[][] mlpBranch = mlpLinearOut(mlpActivated);
        double[][] output = add(afterAttention, mlpBranch);

        return new BlockTrace(normalizedInput, attention.weights(), attentionBranch, afterAttention,
                normalizedForMlp, mlpPreActivation, mlpActivated, mlpBranch, output);
    }

    private static double[][] layerNorm(double[][] values, double[] gamma, double[] beta) {
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
            double inverseStandardDeviation = 1.0 / Math.sqrt(variance + LAYER_NORM_EPSILON);

            for (int channel = 0; channel < EMBEDDING_SIZE; channel++) {
                normalized[position][channel] = (values[position][channel] - mean)
                        * inverseStandardDeviation * gamma[channel] + beta[channel];
            }
        }
        return normalized;
    }

    private static double[][] project(double[][] values, double[][] weights) {
        double[][] projected = new double[SEQUENCE_LENGTH][EMBEDDING_SIZE];
        for (int position = 0; position < SEQUENCE_LENGTH; position++) {
            for (int outputChannel = 0; outputChannel < EMBEDDING_SIZE; outputChannel++) {
                for (int inputChannel = 0; inputChannel < EMBEDDING_SIZE; inputChannel++) {
                    projected[position][outputChannel] += values[position][inputChannel]
                            * weights[outputChannel][inputChannel];
                }
            }
        }
        return projected;
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

    private static double[][] mlpLinearIn(double[][] values) {
        double[][] projected = new double[SEQUENCE_LENGTH][MLP_WIDTH];
        for (int position = 0; position < SEQUENCE_LENGTH; position++) {
            for (int neuron = 0; neuron < MLP_WIDTH; neuron++) {
                projected[position][neuron] = MLP_INPUT_BIAS[neuron];
                for (int channel = 0; channel < EMBEDDING_SIZE; channel++) {
                    projected[position][neuron] += values[position][channel]
                            * MLP_INPUT_WEIGHTS[neuron][channel];
                }
            }
        }
        return projected;
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

    private static double[][] mlpLinearOut(double[][] values) {
        double[][] projected = new double[SEQUENCE_LENGTH][EMBEDDING_SIZE];
        for (int position = 0; position < SEQUENCE_LENGTH; position++) {
            for (int channel = 0; channel < EMBEDDING_SIZE; channel++) {
                projected[position][channel] = MLP_OUTPUT_BIAS[channel];
                for (int neuron = 0; neuron < MLP_WIDTH; neuron++) {
                    projected[position][channel] += values[position][neuron]
                            * MLP_OUTPUT_WEIGHTS[channel][neuron];
                }
            }
        }
        return projected;
    }

    private static double[][] add(double[][] left, double[][] right) {
        double[][] result = new double[SEQUENCE_LENGTH][EMBEDDING_SIZE];
        for (int position = 0; position < SEQUENCE_LENGTH; position++) {
            for (int channel = 0; channel < EMBEDDING_SIZE; channel++) {
                result[position][channel] = left[position][channel] + right[position][channel];
            }
        }
        return result;
    }

    private static boolean residualEquationsHold(double[][] input, BlockTrace trace) {
        for (int position = 0; position < SEQUENCE_LENGTH; position++) {
            for (int channel = 0; channel < EMBEDDING_SIZE; channel++) {
                if (!close(trace.afterAttention()[position][channel], input[position][channel]
                        + trace.attentionBranch()[position][channel])
                        || !close(trace.output()[position][channel], trace.afterAttention()[position][channel]
                        + trace.mlpBranch()[position][channel])
                        || !Double.isFinite(trace.output()[position][channel])) {
                    return false;
                }
            }
        }
        return true;
    }

    private static boolean close(double left, double right) {
        return Math.abs(left - right) < 1e-12;
    }

    private static void printEmbeddingMatrix(String label, double[][] values) {
        System.out.printf("%s (%d x %d):%n", label, values.length, values[0].length);
        printRows(values);
    }

    private static void printAttentionMatrix(String label, double[][] values) {
        System.out.printf("%s (%d x %d):%n", label, values.length, values[0].length);
        printRows(values);
    }

    private static void printHiddenMatrix(String label, double[][] values) {
        System.out.printf("%s (%d x %d):%n", label, values.length, values[0].length);
        printRows(values);
    }

    private static void printRows(double[][] values) {
        for (double[] row : values) {
            System.out.print("  [");
            for (int column = 0; column < row.length; column++) {
                if (column > 0) {
                    System.out.print(", ");
                }
                System.out.print(String.format(Locale.ROOT, "%.4f", row[column]));
            }
            System.out.println("]");
        }
    }

    private record AttentionTrace(double[][] weights, double[][] context) {}

    private record BlockTrace(
            double[][] normalizedInput,
            double[][] attentionWeights,
            double[][] attentionBranch,
            double[][] afterAttention,
            double[][] normalizedForMlp,
            double[][] mlpPreActivation,
            double[][] mlpActivated,
            double[][] mlpBranch,
            double[][] output) {}
}
