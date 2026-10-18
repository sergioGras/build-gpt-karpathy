package learning.karpathy.gpt2;

import java.util.Locale;

public final class ArchitectureLesson {
    private ArchitectureLesson() {}

    public static void main(String[] args) {
        String part = args.length == 1 ? args[0] : "all";
        if (args.length > 1 || !part.matches("all|1|2|3|1-2|2-3|1-3")) {
            System.err.println("Usage: ArchitectureLesson [all|1|2|3|1-2|2-3|1-3]");
            System.exit(1);
        }
        boolean all = part.equals("all");
        boolean dimensions = all || part.equals("1") || part.equals("1-2") || part.equals("1-3");
        boolean attention = all || part.equals("2") || part.equals("1-2")
                || part.equals("2-3") || part.equals("1-3");
        boolean parameterCount = all || part.equals("3") || part.equals("2-3") || part.equals("1-3");

        if (dimensions) {
            showDimensions();
        }
        if (attention) {
            showAttentionWidth();
        }
        if (parameterCount) {
            showParameterCount();
        }
    }

    private static void showDimensions() {
        ModelConfig tutorial1 = ModelConfig.tutorial1Tiny();
        ModelConfig gpt2 = ModelConfig.gpt2();
        System.out.println("Step 1: compare model dimensions (V=vocabulary, T=context, C=width)");
        printDimensions("Tutorial 1 tiny model", tutorial1);
        printDimensions("GPT-2 124M", gpt2);
    }

    private static void printDimensions(String name, ModelConfig config) {
        System.out.printf("%-24s V=%d T=%d L=%d H=%d C=%d%n", name,
                config.vocabSize(), config.contextLength(), config.layers(),
                config.heads(), config.embeddingSize());
    }

    private static void showAttentionWidth() {
        ModelConfig config = ModelConfig.gpt2();
        System.out.println("Step 2: split the residual width evenly across heads");
        System.out.printf("head size = C / H = %d / %d = %d channels per head%n",
                config.embeddingSize(), config.heads(), config.headSize());
        System.out.println("At each sequence position: hidden state (768) -> Q, K, V (768 each).");
        System.out.println("Each is then viewed as 12 heads x 64 channels.");
    }

    private static void showParameterCount() {
        ModelConfig config = ModelConfig.gpt2();
        long parameters = config.gpt2ParameterCount();
        ModelConfig paddedConfig = new ModelConfig(50304, config.contextLength(),
                config.layers(), config.heads(), config.embeddingSize());
        long paddedParameters = paddedConfig.gpt2ParameterCount();
        System.out.println("Step 3: estimate GPT-2 parameters without allocating tensors");
        System.out.println("V*C + T*C + L*(12*C*C + 13*C) + 2*C");
        System.out.printf(Locale.ROOT, "= %,d parameters (%.3f million)%n",
                parameters, parameters / 1_000_000.0);
        System.out.println("The output projection shares its weights with token embeddings.");
        System.out.printf(Locale.ROOT, "Training pads output rows to 50,304: %,d (+%,d)%n",
                paddedParameters, paddedParameters - parameters);
    }
}
