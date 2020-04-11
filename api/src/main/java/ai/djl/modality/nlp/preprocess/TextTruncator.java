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
package ai.djl.modality.nlp.preprocess;

import java.util.List;

/** {@link TextProcessor} that truncates text to a maximum size. */
public class TextTruncator implements TextProcessor {

    int maxSize;

    /**
     * Constructs a {@link TextTruncator}.
     *
     * @param maxSize the size to limit the text to
     */
    public TextTruncator(int maxSize) {
        this.maxSize = maxSize;
    }

    @Override
    public List<String> preprocess(List<String> tokens) {
        if (tokens.size() <= maxSize) {
            return tokens;
        }

        return tokens.subList(0, maxSize);
    }
}
