/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.pytorch.zoo.nlp.qa;

import ai.djl.modality.nlp.Vocabulary;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;

/** A PyTorch implementation of Vocabulary. */
@SuppressWarnings("PMD.UseConcurrentHashMap")
public final class PtBertVocabulary implements Vocabulary {

    private Map<Long, String> vocabMap;
    private Map<String, Long> indicesMap;

    private PtBertVocabulary(Map<Long, String> vocabMap, Map<String, Long> indicesMap) {
        this.vocabMap = vocabMap;
        this.indicesMap = indicesMap;
    }

    /**
     * Parses the vocabulary file and create {@code MxBertVocabulary}.
     *
     * @param path the input file path
     * @return an instance of {@code PtBertVocabulary}
     */
    public static PtBertVocabulary parse(Path path) {
        try {
            return parse(Files.newInputStream(path));
        } catch (IOException e) {
            throw new IllegalStateException(e);
        }
    }

    /**
     * Parses the vocabulary file and create {@code PtBertVocabulary}.
     *
     * @param url the input vocabulary file url
     * @return an instance of {@code PtBertVocabulary}
     */
    public static PtBertVocabulary parse(String url) {
        try {
            return parse(new URL(url).openStream());
        } catch (IOException e) {
            throw new IllegalStateException(e);
        }
    }

    /**
     * Parse the vocabulary file and create {@code PtBertVocabulary}.
     *
     * @param is the input InputStream of the vocabulary file
     * @return an instance of {@code PtBertVocabulary}
     */
    public static PtBertVocabulary parse(InputStream is) {
        try (BufferedReader reader =
                new BufferedReader(new InputStreamReader(is, StandardCharsets.UTF_8))) {
            long i = 0;
            Map<Long, String> vocabMap = new LinkedHashMap<>();
            Map<String, Long> indicesMap = new LinkedHashMap<>();
            while (reader.ready()) {
                String line = reader.readLine();
                vocabMap.put(i, line);
                indicesMap.put(line, i);
                i++;
            }
            return new PtBertVocabulary(vocabMap, indicesMap);
        } catch (IOException e) {
            throw new IllegalStateException(e);
        }
    }

    /** {@inheritDoc} */
    @Override
    public long getIndex(String token) {
        return (indicesMap.containsKey(token)) ? indicesMap.get(token) : indicesMap.get("[UNK]");
    }

    /** {@inheritDoc} */
    @Override
    public String getToken(long index) {
        return vocabMap.get(index);
    }
}
