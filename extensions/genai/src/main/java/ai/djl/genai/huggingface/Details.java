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

import java.util.List;

/** A data class represents text generation schema. */
public class Details {

    @SerializedName("finish_reason")
    private String finishReason;

    @SerializedName("generated_tokens")
    private int generatedTokens;

    private String inputs;
    private List<Token> tokens;
    private List<Token> prefill;

    /**
     * Constructs a {@code Details} instance.
     *
     * @param finishReason the finish reason
     * @param generatedTokens the number of generated tokens
     * @param inputs the input text
     * @param tokens the tokens
     * @param prefill the prefill tokens
     */
    public Details(
            String finishReason,
            int generatedTokens,
            String inputs,
            List<Token> tokens,
            List<Token> prefill) {
        this.finishReason = finishReason;
        this.generatedTokens = generatedTokens;
        this.inputs = inputs;
        this.tokens = tokens;
        this.prefill = prefill;
    }

    /**
     * Returns the finish reason.
     *
     * @return the finish reason
     */
    public String getFinishReason() {
        return finishReason;
    }

    /**
     * Returns the number of generated tokens.
     *
     * @return the number of generated tokens
     */
    public int getGeneratedTokens() {
        return generatedTokens;
    }

    /**
     * Returns the input text.
     *
     * @return the input text
     */
    public String getInputs() {
        return inputs;
    }

    /**
     * Returns the tokens details.
     *
     * @return the tokens details
     */
    public List<Token> getTokens() {
        return tokens;
    }

    /**
     * Returns the prefill tokens information.
     *
     * @return the prefill tokens information
     */
    public List<Token> getPrefill() {
        return prefill;
    }
}
