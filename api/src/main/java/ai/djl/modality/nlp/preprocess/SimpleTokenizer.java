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
package ai.djl.modality.nlp.preprocess;

import java.util.Arrays;
import java.util.List;

/**
 * {@code SimpleTokenizer} is an implementation of the {@link Tokenizer} interface that converts
 * sentences into token by splitting them by a given delimiter.
 */
public class SimpleTokenizer implements Tokenizer {

    private String delimiter = " ";

    /**
     * Creates an instance of {@code SimpleTokenizer} with the given delimiter.
     *
     * @param delimiter the delimiter
     */
    public SimpleTokenizer(String delimiter) {
        this.delimiter = delimiter;
    }

    /** Creates an instance of {@code SimpleTokenizer} with the default delimiter. */
    public SimpleTokenizer() {}

    /** {@inheritDoc} */
    @Override
    public List<String> tokenize(String sentence) {
        return Arrays.asList(sentence.split(" "));
    }

    /** {@inheritDoc} */
    @Override
    public String buildSentence(List<String> tokens) {
        return String.join(delimiter, tokens);
    }
}
