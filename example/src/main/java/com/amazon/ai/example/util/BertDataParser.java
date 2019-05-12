package com.amazon.ai.example.util;

import java.io.FileReader;
import java.io.IOException;
import java.util.*;

import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;

/**
 * This is the Utility for pre-processing the data for Bert Model
 * You can use this utility to parse Vocabulary JSON into Java Array and Dictionary,
 * clean and tokenize sentences and pad the text
 */
public class BertDataParser {

    private Map<String, Integer> token2idx;
    private List<String> idx2token;

    /**
     * Parse the Vocabulary to JSON files
     * [PAD], [CLS], [SEP], [MASK], [UNK] are reserved tokens
     * @param jsonFile the filePath of the vocab.json
     * @throws IOException
     */
    public BertDataParser(String jsonFile) throws IOException {
        Gson gson = new Gson();
        token2idx = new HashMap<>();
        idx2token = new LinkedList<>();
        JsonObject jsonObject = gson.fromJson(new FileReader(jsonFile), JsonObject.class);
        JsonArray arr = jsonObject.getAsJsonArray("idx_to_token");
        for (JsonElement element : arr) {
            idx2token.add(element.getAsString());
        }
        JsonObject preMap = jsonObject.getAsJsonObject("token_to_idx");
        for (String key : preMap.keySet()) {
            token2idx.put(key, preMap.get(key).getAsInt());
        }
    }

    /**
     * Tokenize the input, split all kinds of whitespace and
     * Separate the end of sentence symbol: . , ? !
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
                    finalResult.add(item.substring(item.length() -1));
                } else {
                    finalResult.add(item);
                }
            }
        }
        return finalResult;
    }

    /**
     * Pad the tokens to the required length
     * @param tokens input tokens
     * @param padItem things to pad at the end
     * @param num total length after padding
     * @return List of padded tokens
     */
    public <E> List<E> pad(List<E> tokens, E padItem, int num) {
        if (tokens.size() >= num) return tokens;
        List<E> padded = new LinkedList<>(tokens);
        for (int i = 0; i < num - tokens.size(); i++) {
            padded.add(padItem);
        }
        return padded;
    }

    /**
     * Convert tokens to indexes
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
