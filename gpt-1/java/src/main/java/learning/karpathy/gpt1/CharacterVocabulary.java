package learning.karpathy.gpt1;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeSet;

/** Maps the characters in a training text to stable integer IDs. */
public final class CharacterVocabulary {
    private final Map<Character, Integer> characterToId;
    private final List<Character> idToCharacter;

    private CharacterVocabulary(Map<Character, Integer> characterToId, List<Character> idToCharacter) {
        this.characterToId = characterToId;
        this.idToCharacter = idToCharacter;
    }

    public static CharacterVocabulary fromText(String text) {
        TreeSet<Character> uniqueCharacters = new TreeSet<>();
        for (int index = 0; index < text.length(); index++) {
            uniqueCharacters.add(text.charAt(index));
        }

        Map<Character, Integer> characterToId = new HashMap<>();
        List<Character> idToCharacter = new ArrayList<>();
        for (char character : uniqueCharacters) {
            int id = idToCharacter.size();
            characterToId.put(character, id);
            idToCharacter.add(character);
        }

        return new CharacterVocabulary(characterToId, idToCharacter);
    }

    public int size() {
        return idToCharacter.size();
    }

    public String symbolsInIdOrder() {
        StringBuilder symbols = new StringBuilder();
        for (char character : idToCharacter) {
            if (character == '\n') {
                symbols.append("\\n");
            } else {
                symbols.append(character);
            }
        }
        return symbols.toString();
    }

    public int[] encode(String text) {
        int[] tokenIds = new int[text.length()];
        for (int index = 0; index < text.length(); index++) {
            char character = text.charAt(index);
            Integer id = characterToId.get(character);
            if (id == null) {
                throw new IllegalArgumentException("Character is missing from the vocabulary: " + character);
            }
            tokenIds[index] = id;
        }
        return tokenIds;
    }

    public String decode(int[] tokenIds) {
        StringBuilder text = new StringBuilder(tokenIds.length);
        for (int id : tokenIds) {
            if (id < 0 || id >= idToCharacter.size()) {
                throw new IllegalArgumentException("Token ID is outside this vocabulary: " + id);
            }
            text.append(idToCharacter.get(id));
        }
        return text.toString();
    }
}
