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

import ai.djl.modality.nlp.bert.BertVocabulary;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.LinkedHashMap;
import java.util.Map;

/** A PyTorch implementation of BertVocabulary. */
public final class PtBertVocabulary extends BertVocabulary {
    private Map<Long, String> vocabMap;
    private Map<String, Long> indicesMap;

    private PtBertVocabulary(InputStream is) {
        vocabMap = new LinkedHashMap<>();
        indicesMap = new LinkedHashMap<>();
        try (BufferedReader reader =
                new BufferedReader(new InputStreamReader(is, StandardCharsets.UTF_8))) {
            long i = 0;
            while (reader.ready()) {
                String line = reader.readLine();
                vocabMap.put(i, line);
                indicesMap.put(line, i);
                i++;
            }
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
        return new PtBertVocabulary(is);
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
