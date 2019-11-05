/*
 * Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
 * with the License. A copy of the License is located at
 *
 * http://aws.amazon.com/apache2.0/
 *
 * or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
 * OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
 * and limitations under the License.
 */
package ai.djl.mxnet.zoo.nlp.qa;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.annotations.SerializedName;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.Reader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * This is the Utility for pre-processing data for the Bert Model.
 *
 * <p>You can use this utility to parse vocabulary JSON into Java Array and Dictionary, clean and
 * tokenize sentences, and pad the text.
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
     * Parses the Vocabulary to JSON files. [PAD], [CLS], [SEP], [MASK], [UNK] are reserved tokens.
     *
     * @param is the {@code InputStream} for the vocab.json
     * @return an instance of {@code BertDataParser}
     * @throws IllegalStateException if failed read from {@code InputStream}
     */
    public static BertDataParser parse(InputStream is) {
        try (Reader reader = new InputStreamReader(is, StandardCharsets.UTF_8)) {
            return GSON.fromJson(reader, BertDataParser.class);
        } catch (IOException e) {
            throw new IllegalStateException(e);
        }
    }

    /**
     * Tokenizes the input, splits all kinds of whitespace, and separates the end of sentence
     * symbol.
     *
     * @param input the input string
     * @return a list of tokens
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
     * Pads the tokens to the required length.
     *
     * @param <E> the type of the List
     * @param tokens the input tokens
     * @param padItem the things to pad at the end
     * @param num the total length after padding
     * @return a list of padded tokens
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
     * Forms the token types List [0000...1111...000] where all questions are 0 and answers are 1.
     *
     * @param question the question tokens
     * @param answer the answer tokens
     * @param seqLength the sequence length
     * @return a list of tokenTypes
     */
    public static List<Float> getTokenTypes(
            List<String> question, List<String> answer, int seqLength) {
        List<Float> qaEmbedded = new ArrayList<>();
        qaEmbedded = pad(qaEmbedded, 0f, question.size() + 2);
        qaEmbedded.addAll(pad(new ArrayList<>(), 1f, answer.size()));
        return pad(qaEmbedded, 0f, seqLength);
    }

    /**
     * Forms tokens with separation that can be used for BERT.
     *
     * @param question the question tokens
     * @param answer the answer tokens
     * @param seqLength the sequence length
     * @return a list of tokenTypes
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
     * Converts tokens to indexes.
     *
     * @param tokens the input tokens
     * @return a list of indexes
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
     * Converts indexes to tokens.
     *
     * @param indexes the list of indexes
     * @return a list of tokens
     */
    public List<String> idx2token(List<Integer> indexes) {
        List<String> tokens = new ArrayList<>();
        for (int index : indexes) {
            tokens.add(idx2token.get(index));
        }
        return tokens;
    }
}
