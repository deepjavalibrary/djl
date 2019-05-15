package com.amazon.ai.example.util;

import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.annotations.SerializedName;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

/**
 * This is the Utility for pre-processing the data for Bert Model You can use this utility to parse
 * Vocabulary JSON into Java Array and Dictionary, clean and tokenize sentences and pad the text
 */
public class BertDataParser {

    private class VocabularyInfo {
        @SerializedName("token_to_idx")
        private Map<String, Integer> token2idx;
        @SerializedName("idx_to_token")
        private List<String> idx2token;
    }

    private VocabularyInfo vocab;

    /**
     * Parse the Vocabulary to JSON files [PAD], [CLS], [SEP], [MASK], [UNK] are reserved tokens
     *
     * @param jsonFile the filePath of the vocab.json
     * @throws IOException
     */
    public BertDataParser(String jsonFile) throws IOException {
        Gson gson = new Gson();
        vocab = gson.fromJson(Files.newBufferedReader(new File(jsonFile).toPath()), VocabularyInfo.class);
    }

    /**
     * Tokenize the input, split all kinds of whitespace and Separate the end of sentence symbol: .
     * , ? !
     *
     * @param input The input string
     * @return List of tokens
     */
    public List<String> tokenizer(String input) {
        String[] step1 = input.split("\\s+");
        List<String> finalResult = new LinkedList<>();
        for (String item : step1) {
            if (item.length() != 0) {
                if ((item + "a").split("[.,?!]+").length > 1) {
                    finalResult.add(item.substring(0, item.length() - 1));
                    finalResult.add(item.substring(item.length() - 1));
                } else {
                    finalResult.add(item);
                }
            }
        }
        return finalResult;
    }

    /**
     * Pad the tokens to the required length
     *
     * @param tokens input tokens
     * @param padItem things to pad at the end
     * @param num total length after padding
     * @return List of padded tokens
     */
    public <E> List<E> pad(List<E> tokens, E padItem, int num) {
        if (tokens.size() >= num) {
            return tokens;
        }
        List<E> padded = new LinkedList<>(tokens);
        for (int i = 0; i < num - tokens.size(); i++) {
            padded.add(padItem);
        }
        return padded;
    }

    /**
     * Form the token types List [0000...1111...000]
     * All questions are 0 and Answers are 1
     * @param question question tokens
     * @param answer answer tokens
     * @param seqLength sequence length
     * @return List of tokenTypes
     */
    public List<Float> getTokenTypes(List<String> question, List<String> answer, int seqLength) {
        List<Float> qaEmbedded = new ArrayList<>();
        qaEmbedded = pad(qaEmbedded, 0f, question.size() + 2);
        qaEmbedded.addAll(pad(new ArrayList<>(), 1f, answer.size()));
        return pad(qaEmbedded, 0f, seqLength);
    }

    /**
     * Form tokens with separation that can be used for BERT
     * @param question question tokens
     * @param answer answer tokens
     * @param seqLength sequence length
     * @return List of tokenTypes
     */
    public List<String> formTokens(List<String> question, List<String> answer, int seqLength) {
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
            if (vocab.token2idx.containsKey(token)) {
                indexes.add(vocab.token2idx.get(token));
            } else {
                indexes.add(vocab.token2idx.get("[UNK]"));
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
            tokens.add(vocab.idx2token.get(index));
        }
        return tokens;
    }
}
