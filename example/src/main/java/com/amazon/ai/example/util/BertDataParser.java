package com.amazon.ai.example.util;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.annotations.SerializedName;
import java.io.File;
import java.io.IOException;
import java.io.Reader;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * This is the Utility for pre-processing the data for Bert Model You can use this utility to parse
 * Vocabulary JSON into Java Array and Dictionary, clean and tokenize sentences and pad the text
 */
@SuppressWarnings("unused")
public class BertDataParser {

    private static final Gson GSON = new GsonBuilder().create();
    private static final Pattern PATTERN = Pattern.compile("(\\S+?)([.,?!])?(\\s+|$)");

    @SerializedName("token_to_idx")
    private Map<String, Integer> token2idx;

    @SerializedName("idx_to_token")
    private List<String> idx2token;

    /**
     * Parse the Vocabulary to JSON files [PAD], [CLS], [SEP], [MASK], [UNK] are reserved tokens.
     *
     * @param jsonFile the file of the vocab.json
     * @return instance of <code>BertDataParser</code>
     * @throws IOException if failed read from file
     */
    public static BertDataParser parse(File jsonFile) throws IOException {
        try (Reader reader = Files.newBufferedReader(jsonFile.toPath())) {
            return GSON.fromJson(reader, BertDataParser.class);
        }
    }

    /**
     * Tokenize the input, split all kinds of whitespace and Separate the end of sentence symbol: .
     * , ? !
     *
     * @param input The input string
     * @return List of tokens
     */
    public static List<String> tokenizer(String input) {
        List<String> ret = new LinkedList<>();

        Matcher m = PATTERN.matcher(input);
        while (m.find()) {
            ret.add(m.group(1));
            String token = m.group(2);
            if (token != null) {
                ret.add(token);
            }
        }

        return ret;
    }

    /**
     * Pad the tokens to the required length.
     *
     * @param <E> the type of the List
     * @param tokens input tokens
     * @param padItem things to pad at the end
     * @param num total length after padding
     * @return List of padded tokens
     */
    public static <E> List<E> pad(List<E> tokens, E padItem, int num) {
        if (tokens.size() >= num) {
            return tokens;
        }
        List<E> padded = new ArrayList<>(num);
        padded.addAll(tokens);
        for (int i = tokens.size(); i < num; ++i) {
            padded.add(padItem);
        }
        return padded;
    }

    /**
     * Form the token types List [0000...1111...000] All questions are 0 and Answers are 1
     *
     * @param question question tokens
     * @param answer answer tokens
     * @param seqLength sequence length
     * @return List of tokenTypes
     */
    public static List<Float> getTokenTypes(
            List<String> question, List<String> answer, int seqLength) {
        List<Float> qaEmbedded = new ArrayList<>();
        qaEmbedded = pad(qaEmbedded, 0f, question.size() + 2);
        qaEmbedded.addAll(pad(new ArrayList<>(), 1f, answer.size()));
        return pad(qaEmbedded, 0f, seqLength);
    }

    /**
     * Form tokens with separation that can be used for BERT
     *
     * @param question question tokens
     * @param answer answer tokens
     * @param seqLength sequence length
     * @return List of tokenTypes
     */
    public static List<String> formTokens(
            List<String> question, List<String> answer, int seqLength) {
        // make BERT pre-processing standard
        List<String> tokens = new ArrayList<>(question);
        tokens.add("[SEP]");
        tokens.add(0, "[CLS]");
        answer.add("[SEP]");
        tokens.addAll(answer);
        tokens.add("[SEP]");
        return pad(tokens, "[PAD]", seqLength);
    }

    /**
     * Convert tokens to indexes
     *
     * @param tokens input tokens
     * @return List of indexes
     */
    public List<Integer> token2idx(List<String> tokens) {
        List<Integer> indexes = new ArrayList<>();
        for (String token : tokens) {
            if (token2idx.containsKey(token)) {
                indexes.add(token2idx.get(token));
            } else {
                indexes.add(token2idx.get("[UNK]"));
            }
        }
        return indexes;
    }

    /**
     * Convert indexes to tokens
     *
     * @param indexes List of indexes
     * @return List of tokens
     */
    public List<String> idx2token(List<Integer> indexes) {
        List<String> tokens = new ArrayList<>();
        for (int index : indexes) {
            tokens.add(idx2token.get(index));
        }
        return tokens;
    }
}
