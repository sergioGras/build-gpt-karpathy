package learning.karpathy.gpt1;

import java.util.Locale;

/** Connects embeddings, Transformer blocks, vocabulary logits, loss, and generation. */
public final class DecoderLanguageModelLesson {
    private static final int BATCH_SIZE = 2;
    private static final int BLOCK_SIZE = 4;
    private static final int VOCAB_SIZE = 5;
    private static final int CHANNELS = 4;
    private static final int HEADS = 2;
    private static final int HEAD_SIZE = CHANNELS / HEADS;
    private static final int FEED_FORWARD_SIZE = 4 * CHANNELS;
    private static final int LAYERS = 2;
    private static final int GENERATED_TOKENS = 5;
    private static final double EPSILON = 1e-5;

    private static final String[] VOCABULARY = { "a", "b", "c", "d", "e" };
    private static final int[][] TOKEN_IDS = {
            { 0, 1, 0, 2 },
            { 1, 2, 3, 0 }
    };
    private static final int[][] TARGET_IDS = {
            { 1, 0, 2, 3 },
            { 2, 3, 0, 1 }
    };
    private static final double[] SAMPLE_THRESHOLDS = { 0.19, 0.47, 0.81, 0.33, 0.68 };

    private DecoderLanguageModelLesson() {
    }

    public static void main(String[] args) {
        Locale.setDefault(Locale.ROOT);
        int[] steps = parseSteps(args);
        ModelOutput output = forward(TOKEN_IDS, BLOCK_SIZE);

        if (steps[0] <= 1) {
            System.out.println("Step 1 - Embed token IDs and add position vectors");
            System.out.println("Vocabulary IDs: a=0, b=1, c=2, d=3, e=4");
            System.out.printf("Tiny model: vocabulary=%d, model width=%d, context=%d, layers=%d, heads=%d%n",
                    VOCAB_SIZE, CHANNELS, BLOCK_SIZE, LAYERS, HEADS);
            printTensor("Input to the Transformer blocks", output.embeddingOutput(), BLOCK_SIZE);
        }

        if (steps[0] <= 2 && steps[1] >= 2) {
            System.out.println("Step 2 - Pass representations through two pre-LayerNorm blocks");
            System.out.println("Each block adds causal multi-head attention, then a position-wise feed-forward network.");
            for (int layer = 0; layer < LAYERS; layer++) {
                printTensor("Output after block " + (layer + 1), output.blockOutputs()[layer], BLOCK_SIZE);
            }
        }

        if (steps[0] <= 3 && steps[1] >= 3) {
            System.out.println("Step 3 - Project each final vector to next-token logits");
            printIds("Next-token target IDs", TARGET_IDS, BLOCK_SIZE);
            printLogits(output.logits(), BLOCK_SIZE);
            System.out.printf("Mean next-token cross-entropy on the tiny target batch: %.6f%n",
                    crossEntropy(output.logits(), BLOCK_SIZE));
            System.out.println("The output width is the vocabulary size; these fixed weights have not been trained.");
        }

        if (steps[1] >= 4) {
            int[] generated = generate();
            System.out.println("Step 4 - Generate one token at a time");
            System.out.println("Each step uses the last position's logits; the context is cropped to the four-position limit.");
            System.out.printf("Prompt: %s%s%nGenerated sequence: ",
                    VOCABULARY[generated[0]], VOCABULARY[generated[1]]);
            for (int tokenId : generated) {
                System.out.print(VOCABULARY[tokenId]);
            }
            System.out.println();
            System.out.println("Fixed cumulative-probability thresholds make this untrained demo repeatable across languages.");
        }
    }

    private static ModelOutput forward(int[][] tokenIds, int timeSteps) {
        double[][] tokenTable = new double[VOCAB_SIZE][CHANNELS];
        double[][] positionTable = new double[BLOCK_SIZE][CHANNELS];
        initializeEmbeddings(tokenTable, positionTable);

        double[][][] embeddingOutput = new double[BATCH_SIZE][BLOCK_SIZE][CHANNELS];
        for (int batch = 0; batch < BATCH_SIZE; batch++) {
            for (int time = 0; time < timeSteps; time++) {
                for (int channel = 0; channel < CHANNELS; channel++) {
                    embeddingOutput[batch][time][channel] =
                            tokenTable[tokenIds[batch][time]][channel] + positionTable[time][channel];
                }
            }
        }

        double[][][][] blockOutputs = new double[LAYERS][BATCH_SIZE][BLOCK_SIZE][CHANNELS];
        double[][][] current = copyTensor(embeddingOutput, timeSteps);
        for (int layer = 0; layer < LAYERS; layer++) {
            double[][][] normalized = layerNorm(current, timeSteps);
            double[][][] attentionOutput = multiHeadAttention(normalized, layer, timeSteps);
            double[][][] afterAttention = add(current, attentionOutput, timeSteps);
            double[][][] normalizedAgain = layerNorm(afterAttention, timeSteps);
            double[][][] feedForwardOutput = feedForward(normalizedAgain, layer, timeSteps);
            blockOutputs[layer] = add(afterAttention, feedForwardOutput, timeSteps);
            current = copyTensor(blockOutputs[layer], timeSteps);
        }

        double[][][] normalizedFinal = layerNorm(current, timeSteps);
        double[][][] logits = new double[BATCH_SIZE][BLOCK_SIZE][VOCAB_SIZE];
        for (int batch = 0; batch < BATCH_SIZE; batch++) {
            for (int time = 0; time < timeSteps; time++) {
                for (int token = 0; token < VOCAB_SIZE; token++) {
                    double value = deterministicBias(42, token);
                    for (int channel = 0; channel < CHANNELS; channel++) {
                        value += normalizedFinal[batch][time][channel]
                                * deterministicWeight(41, token, channel, 11, 0.05);
                    }
                    logits[batch][time][token] = value;
                }
            }
        }
        return new ModelOutput(embeddingOutput, blockOutputs, logits);
    }

    private static double[][][] multiHeadAttention(double[][][] input, int layer, int timeSteps) {
        double[][][] concatenated = new double[BATCH_SIZE][BLOCK_SIZE][CHANNELS];
        for (int head = 0; head < HEADS; head++) {
            int seed = layer * 10 + head * 3;
            double[][][] queries = projectHead(input, seed + 1, timeSteps);
            double[][][] keys = projectHead(input, seed + 2, timeSteps);
            double[][][] values = projectHead(input, seed + 3, timeSteps);

            for (int batch = 0; batch < BATCH_SIZE; batch++) {
                for (int time = 0; time < timeSteps; time++) {
                    double[] scores = new double[time + 1];
                    double maximum = Double.NEGATIVE_INFINITY;
                    for (int keyTime = 0; keyTime <= time; keyTime++) {
                        double score = 0.0;
                        for (int channel = 0; channel < HEAD_SIZE; channel++) {
                            score += queries[batch][time][channel] * keys[batch][keyTime][channel];
                        }
                        scores[keyTime] = score / Math.sqrt(HEAD_SIZE);
                        maximum = Math.max(maximum, scores[keyTime]);
                    }
                    double total = 0.0;
                    for (int keyTime = 0; keyTime <= time; keyTime++) {
                        scores[keyTime] = Math.exp(scores[keyTime] - maximum);
                        total += scores[keyTime];
                    }
                    for (int channel = 0; channel < HEAD_SIZE; channel++) {
                        double value = 0.0;
                        for (int keyTime = 0; keyTime <= time; keyTime++) {
                            value += scores[keyTime] / total * values[batch][keyTime][channel];
                        }
                        concatenated[batch][time][head * HEAD_SIZE + channel] = value;
                    }
                }
            }
        }

        double[][][] output = new double[BATCH_SIZE][BLOCK_SIZE][CHANNELS];
        int projectionSeed = layer * 10 + 7;
        int biasSeed = layer * 10 + 8;
        for (int batch = 0; batch < BATCH_SIZE; batch++) {
            for (int time = 0; time < timeSteps; time++) {
                for (int outputChannel = 0; outputChannel < CHANNELS; outputChannel++) {
                    double value = deterministicBias(biasSeed, outputChannel);
                    for (int inputChannel = 0; inputChannel < CHANNELS; inputChannel++) {
                        value += concatenated[batch][time][inputChannel]
                                * deterministicWeight(projectionSeed, outputChannel, inputChannel, 11, 0.05);
                    }
                    output[batch][time][outputChannel] = value;
                }
            }
        }
        return output;
    }

    private static double[][][] projectHead(double[][][] input, int seed, int timeSteps) {
        double[][][] projected = new double[BATCH_SIZE][BLOCK_SIZE][HEAD_SIZE];
        for (int batch = 0; batch < BATCH_SIZE; batch++) {
            for (int time = 0; time < timeSteps; time++) {
                for (int outputChannel = 0; outputChannel < HEAD_SIZE; outputChannel++) {
                    for (int inputChannel = 0; inputChannel < CHANNELS; inputChannel++) {
                        projected[batch][time][outputChannel] += input[batch][time][inputChannel]
                                * deterministicWeight(seed, outputChannel, inputChannel, 11, 0.08);
                    }
                }
            }
        }
        return projected;
    }

    private static double[][][] feedForward(double[][][] input, int layer, int timeSteps) {
        double[][][] output = new double[BATCH_SIZE][BLOCK_SIZE][CHANNELS];
        int firstSeed = layer * 13 + 9;
        int firstBiasSeed = layer * 13 + 10;
        int secondSeed = layer * 13 + 11;
        int secondBiasSeed = layer * 13 + 12;

        for (int batch = 0; batch < BATCH_SIZE; batch++) {
            for (int time = 0; time < timeSteps; time++) {
                double[] hidden = new double[FEED_FORWARD_SIZE];
                for (int hiddenChannel = 0; hiddenChannel < FEED_FORWARD_SIZE; hiddenChannel++) {
                    double value = deterministicBias(firstBiasSeed, hiddenChannel);
                    for (int inputChannel = 0; inputChannel < CHANNELS; inputChannel++) {
                        value += input[batch][time][inputChannel]
                                * deterministicWeight(firstSeed, hiddenChannel, inputChannel, 11, 0.04);
                    }
                    hidden[hiddenChannel] = Math.max(0.0, value);
                }
                for (int outputChannel = 0; outputChannel < CHANNELS; outputChannel++) {
                    double value = deterministicBias(secondBiasSeed, outputChannel);
                    for (int hiddenChannel = 0; hiddenChannel < FEED_FORWARD_SIZE; hiddenChannel++) {
                        value += hidden[hiddenChannel]
                                * deterministicWeight(secondSeed, outputChannel, hiddenChannel, 11, 0.03);
                    }
                    output[batch][time][outputChannel] = value;
                }
            }
        }
        return output;
    }

    private static double[][][] layerNorm(double[][][] input, int timeSteps) {
        double[][][] output = new double[BATCH_SIZE][BLOCK_SIZE][CHANNELS];
        for (int batch = 0; batch < BATCH_SIZE; batch++) {
            for (int time = 0; time < timeSteps; time++) {
                double mean = 0.0;
                for (int channel = 0; channel < CHANNELS; channel++) {
                    mean += input[batch][time][channel];
                }
                mean /= CHANNELS;
                double variance = 0.0;
                for (int channel = 0; channel < CHANNELS; channel++) {
                    double centered = input[batch][time][channel] - mean;
                    variance += centered * centered;
                }
                variance /= CHANNELS;
                for (int channel = 0; channel < CHANNELS; channel++) {
                    output[batch][time][channel] =
                            (input[batch][time][channel] - mean) / Math.sqrt(variance + EPSILON);
                }
            }
        }
        return output;
    }

    private static double[][][] add(double[][][] left, double[][][] right, int timeSteps) {
        double[][][] result = new double[BATCH_SIZE][BLOCK_SIZE][CHANNELS];
        for (int batch = 0; batch < BATCH_SIZE; batch++) {
            for (int time = 0; time < timeSteps; time++) {
                for (int channel = 0; channel < CHANNELS; channel++) {
                    result[batch][time][channel] = left[batch][time][channel] + right[batch][time][channel];
                }
            }
        }
        return result;
    }

    private static double[][][] copyTensor(double[][][] source, int timeSteps) {
        double[][][] copy = new double[BATCH_SIZE][BLOCK_SIZE][CHANNELS];
        for (int batch = 0; batch < BATCH_SIZE; batch++) {
            for (int time = 0; time < timeSteps; time++) {
                System.arraycopy(source[batch][time], 0, copy[batch][time], 0, CHANNELS);
            }
        }
        return copy;
    }

    private static void initializeEmbeddings(double[][] tokenTable, double[][] positionTable) {
        for (int token = 0; token < VOCAB_SIZE; token++) {
            for (int channel = 0; channel < CHANNELS; channel++) {
                tokenTable[token][channel] = ((token * 5 + channel * 3 + 1) % 11 - 5) * 0.1;
            }
        }
        for (int position = 0; position < BLOCK_SIZE; position++) {
            for (int channel = 0; channel < CHANNELS; channel++) {
                positionTable[position][channel] = ((position * 7 + channel * 2 + 2) % 9 - 4) * 0.1;
            }
        }
    }

    private static double deterministicWeight(
            int seed, int outputChannel, int inputChannel, int modulus, double scale) {
        int pattern = (seed * 5 + (outputChannel + 1) * (inputChannel + 2)) % modulus;
        return (pattern - modulus / 2) * scale;
    }

    private static double deterministicBias(int seed, int channel) {
        return ((seed + channel * 2) % 5 - 2) * 0.01;
    }

    private static double crossEntropy(double[][][] logits, int timeSteps) {
        double totalLoss = 0.0;
        for (int batch = 0; batch < BATCH_SIZE; batch++) {
            for (int time = 0; time < timeSteps; time++) {
                double maximum = logits[batch][time][0];
                for (int token = 1; token < VOCAB_SIZE; token++) {
                    maximum = Math.max(maximum, logits[batch][time][token]);
                }
                double exponentialSum = 0.0;
                for (int token = 0; token < VOCAB_SIZE; token++) {
                    exponentialSum += Math.exp(logits[batch][time][token] - maximum);
                }
                int target = TARGET_IDS[batch][time];
                totalLoss += maximum + Math.log(exponentialSum) - logits[batch][time][target];
            }
        }
        return totalLoss / (BATCH_SIZE * timeSteps);
    }

    private static int[] generate() {
        int[] generated = new int[2 + GENERATED_TOKENS];
        generated[0] = 0;
        generated[1] = 1;
        int generatedCount = 2;
        for (int step = 0; step < GENERATED_TOKENS; step++) {
            int contextLength = Math.min(generatedCount, BLOCK_SIZE);
            int start = generatedCount - contextLength;
            int[][] context = new int[BATCH_SIZE][BLOCK_SIZE];
            for (int time = 0; time < contextLength; time++) {
                context[0][time] = generated[start + time];
            }
            ModelOutput output = forward(context, contextLength);
            generated[generatedCount++] = sampleToken(
                    output.logits()[0][contextLength - 1], SAMPLE_THRESHOLDS[step]);
        }
        return generated;
    }

    private static int sampleToken(double[] logits, double threshold) {
        double maximum = logits[0];
        for (int token = 1; token < VOCAB_SIZE; token++) {
            maximum = Math.max(maximum, logits[token]);
        }
        double total = 0.0;
        double[] probabilities = new double[VOCAB_SIZE];
        for (int token = 0; token < VOCAB_SIZE; token++) {
            probabilities[token] = Math.exp(logits[token] - maximum);
            total += probabilities[token];
        }
        double cumulative = 0.0;
        for (int token = 0; token < VOCAB_SIZE; token++) {
            cumulative += probabilities[token] / total;
            if (threshold < cumulative) {
                return token;
            }
        }
        return VOCAB_SIZE - 1;
    }

    private static void printTensor(String label, double[][][] values, int timeSteps) {
        System.out.printf("%s, shape [batch, time, channels] = [%d, %d, %d]:%n",
                label, BATCH_SIZE, timeSteps, CHANNELS);
        for (int batch = 0; batch < BATCH_SIZE; batch++) {
            System.out.printf("  batch %d%n", batch);
            for (int time = 0; time < timeSteps; time++) {
                System.out.printf("    t=%d [", time);
                for (int channel = 0; channel < CHANNELS; channel++) {
                    System.out.printf("%s%.4f", channel == 0 ? "" : ", ", values[batch][time][channel]);
                }
                System.out.println("]");
            }
        }
    }

    private static void printLogits(double[][][] logits, int timeSteps) {
        System.out.printf("Logits, shape [batch, time, vocab] = [%d, %d, %d] (batch 0):%n",
                BATCH_SIZE, timeSteps, VOCAB_SIZE);
        for (int time = 0; time < timeSteps; time++) {
            System.out.printf("  t=%d [", time);
            for (int token = 0; token < VOCAB_SIZE; token++) {
                System.out.printf("%s%.4f", token == 0 ? "" : ", ", logits[0][time][token]);
            }
            System.out.println("]");
        }
    }

    private static void printIds(String label, int[][] ids, int timeSteps) {
        System.out.printf("%s, shape [batch, time] = [%d, %d]:%n", label, BATCH_SIZE, timeSteps);
        for (int batch = 0; batch < BATCH_SIZE; batch++) {
            System.out.printf("  batch %d [", batch);
            for (int time = 0; time < timeSteps; time++) {
                System.out.printf("%s%d", time == 0 ? "" : ", ", ids[batch][time]);
            }
            System.out.println("]");
        }
    }

    private static int[] parseSteps(String[] args) {
        if (args.length > 1) {
            throw new IllegalArgumentException(usage());
        }
        String part = args.length == 0 ? "all" : args[0];
        return switch (part) {
            case "1" -> new int[] { 1, 1 };
            case "2" -> new int[] { 2, 2 };
            case "3" -> new int[] { 3, 3 };
            case "4" -> new int[] { 4, 4 };
            case "1-2" -> new int[] { 1, 2 };
            case "1-3" -> new int[] { 1, 3 };
            case "2-4" -> new int[] { 2, 4 };
            case "3-4" -> new int[] { 3, 4 };
            case "all", "1-4" -> new int[] { 1, 4 };
            default -> throw new IllegalArgumentException(usage());
        };
    }

    private static String usage() {
        return "Usage: DecoderLanguageModelLesson [all|1|2|3|4|1-2|1-3|1-4|2-4|3-4]";
    }

    private record ModelOutput(double[][][] embeddingOutput, double[][][][] blockOutputs, double[][][] logits) {
    }
}
