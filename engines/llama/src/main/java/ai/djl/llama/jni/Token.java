/*
 * Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.llama.jni;

import java.nio.charset.StandardCharsets;
import java.util.Map;

/** The output token class. */
public final class Token {

    private int token;
    private String text;
    private Map<Integer, Float> probabilities;
    transient long count;
    transient long pos;
    transient boolean hasNext;

    /**
     * Constructs a new {@code Token} instance.
     *
     * @param token the token id
     * @param generated the token text
     * @param probabilities the token probabilities
     * @param count the generated token count
     * @param pos the token index
     * @param hasNext has more tokens
     */
    public Token(
            int token,
            byte[] generated,
            Map<Integer, Float> probabilities,
            long count,
            long pos,
            boolean hasNext) {
        this.token = token;
        this.text = new String(generated, StandardCharsets.UTF_8);
        this.probabilities = probabilities;
        this.count = count;
        this.pos = pos;
        this.hasNext = hasNext;
    }

    /**
     * Returns the token id.
     *
     * @return the token id
     */
    public int getToken() {
        return token;
    }

    /**
     * Returns the token text.
     *
     * @return the token text
     */
    public String getText() {
        return text;
    }

    /**
     * Returns the token probabilities.
     *
     * @return the token probabilities
     */
    public Map<Integer, Float> getProbabilities() {
        return probabilities;
    }

    /** {@inheritDoc} */
    @Override
    public String toString() {
        return text;
    }
}
