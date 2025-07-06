/*
 * Copyright 2025 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.genai.huggingface;

import com.google.gson.annotations.SerializedName;

/** A data class represents text generation schema. */
public class Token {

    private int id;
    private String text;

    @SerializedName("log_prob")
    private double logprob;

    /**
     * Constructs a {@code Token} instance.
     *
     * @param id the token id
     * @param text the text
     * @param logprob the log probability
     */
    public Token(int id, String text, double logprob) {
        this.id = id;
        this.text = text;
        this.logprob = logprob;
    }

    /**
     * Returns the token id.
     *
     * @return the token id
     */
    public int getId() {
        return id;
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
     * Returns the log probability.
     *
     * @return the log probability
     */
    public double getLogprob() {
        return logprob;
    }
}
