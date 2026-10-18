package learning.karpathy.gpt2;

public final class Main {
    private Main() {}

    public static void main(String[] args) {
        String preset = args.length == 1 ? args[0] : "tiny";
        if (args.length > 1 || (!preset.equals("tiny") && !preset.equals("gpt2"))) {
            System.err.println("Usage: Main [tiny|gpt2]");
            System.exit(1);
        }
        ModelConfig config = preset.equals("gpt2") ? ModelConfig.gpt2() : ModelConfig.tiny();
        System.out.printf("GPT-2 Java scaffold (%s)%n", preset);
        System.out.printf("vocab=%d context=%d layers=%d heads=%d embedding=%d head_size=%d%n",
                config.vocabSize(), config.contextLength(), config.layers(),
                config.heads(), config.embeddingSize(), config.headSize());
        System.out.println("Configuration inspection only; no model weights are allocated.");
    }
}
