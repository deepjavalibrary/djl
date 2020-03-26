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

import java.util.ArrayList;
import java.util.List;
import java.util.StringTokenizer;

/** {@code PunctuationSeparator} separates punction into a separate token. */
public class PunctuationSeparator implements TextProcessor {

    private static final String DEFAULT_PUNCTUATIONS = ".,!?";

    private String punctuations;

    /**
     * Creates a {@link TextProcessor} that separates the given punctuations into distinct tokens in
     * the text.
     *
     * @param punctuations the punctuations to be separated
     */
    public PunctuationSeparator(String punctuations) {
        this.punctuations = punctuations;
    }

    /**
     * Creates a {@link TextProcessor} that separates the given punctuations into distinct tokens in
     * the text.
     */
    public PunctuationSeparator() {
        this(DEFAULT_PUNCTUATIONS);
    }

    /** {@inheritDoc} */
    @Override
    public List<String> preprocess(List<String> tokens) {
        List<String> list = new ArrayList<>();
        for (String token : tokens) {
            StringTokenizer tokenizer = new StringTokenizer(token, punctuations, true);
            while (tokenizer.hasMoreElements()) {
                String element = tokenizer.nextToken();
                if (!element.isEmpty()) {
                    list.add(element);
                }
            }
        }
        return list;
    }
}
