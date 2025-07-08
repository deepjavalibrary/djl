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
public class GenerationOutput {

    @SerializedName("generated_text")
    private String generatedText;

    private Details details;
    private Token token;

    /**
     * Constructs a {@code GenerationOutput} instance.
     *
     * @param generatedText the generated text
     * @param details the details
     * @param token the token
     */
    public GenerationOutput(String generatedText, Details details, Token token) {
        this.generatedText = generatedText;
        this.details = details;
        this.token = token;
    }

    /**
     * Returns the generated text.
     *
     * @return the generated text
     */
    public String getGeneratedText() {
        return generatedText;
    }

    /**
     * Returns the details.
     *
     * @return the details
     */
    public Details getDetails() {
        return details;
    }

    /**
     * Returns the token.
     *
     * @return the token
     */
    public Token getToken() {
        return token;
    }
}
