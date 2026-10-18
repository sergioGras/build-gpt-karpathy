package learning.karpathy.gpt1;

import java.io.IOException;
import java.util.Arrays;

/** Runnable companion to lesson 2: turn characters into integer token IDs. */
public final class CharacterTokenizationLesson {
    private CharacterTokenizationLesson() {
    }

    public static void main(String[] args) throws IOException {
        int[] steps = parseSteps(args);
        TinyShakespeareDataset dataset = TinyShakespeareDataset.load(TinyShakespeareDataset.findDefaultPath());
        String text = dataset.text();
        CharacterVocabulary vocabulary = CharacterVocabulary.fromText(text);

        if (steps[0] <= 1) {
            System.out.println("Step 1 - Build the character vocabulary");
            System.out.printf("Vocabulary size: %d%n", vocabulary.size());
            System.out.printf("Characters in ID order: \"%s\"%n", vocabulary.symbolsInIdOrder());
        }

        if (steps[0] <= 2 && steps[1] >= 2) {
            String sample = "hii there";
            int[] sampleIds = vocabulary.encode(sample);
            String decodedSample = vocabulary.decode(sampleIds);
            System.out.println("Step 2 - Encode and decode a toy example");
            System.out.printf("Input text: %s%n", sample);
            System.out.printf("Token IDs: %s%n", Arrays.toString(sampleIds));
            System.out.printf("Decoded text: %s%n", decodedSample);
            System.out.printf("Round trip matches: %s%n", sample.equals(decodedSample));
        }

        if (steps[1] >= 3) {
            int[] tokenIds = vocabulary.encode(text);
            int previewLength = Math.min(100, tokenIds.length);
            boolean roundTripMatches = text.equals(vocabulary.decode(tokenIds));
            System.out.println("Step 3 - Encode the full corpus");
            System.out.printf("Token count: %d%n", tokenIds.length);
            System.out.printf("First %d token IDs: %s%n", previewLength,
                    Arrays.toString(Arrays.copyOf(tokenIds, previewLength)));
            System.out.printf("Full text round trip matches: %s%n", roundTripMatches);
        }
    }

    private static int[] parseSteps(String[] args) {
        if (args.length > 1) {
            throw new IllegalArgumentException("Usage: CharacterTokenizationLesson [all|1|2|3|1-2|1-3]");
        }

        String part = args.length == 0 ? "all" : args[0];
        return switch (part) {
            case "1" -> new int[] { 1, 1 };
            case "2" -> new int[] { 2, 2 };
            case "3" -> new int[] { 3, 3 };
            case "1-2" -> new int[] { 1, 2 };
            case "all", "1-3" -> new int[] { 1, 3 };
            default -> throw new IllegalArgumentException(
                    "Usage: CharacterTokenizationLesson [all|1|2|3|1-2|1-3]");
        };
    }
}
