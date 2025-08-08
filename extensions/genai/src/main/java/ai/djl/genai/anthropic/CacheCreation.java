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
package ai.djl.genai.anthropic;

import com.google.gson.annotations.SerializedName;

/** A data class represents Anthropic schema. */
@SuppressWarnings("MissingJavadocMethod")
public class CacheCreation {

    @SerializedName("ephemeral_1h_input_tokens")
    private int ephemeral1hInputTokens;

    @SerializedName("ephemeral_5m_input_tokens")
    private int ephemeral5mInputTokens;

    CacheCreation(Builder builder) {
        this.ephemeral1hInputTokens = builder.ephemeral1hInputTokens;
        this.ephemeral5mInputTokens = builder.ephemeral5mInputTokens;
    }

    public int getEphemeral1hInputTokens() {
        return ephemeral1hInputTokens;
    }

    public int getEphemeral5mInputTokens() {
        return ephemeral5mInputTokens;
    }

    /**
     * Creates a builder to build a {@code CacheCreation}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /** The builder for {@code CacheCreation}. */
    public static final class Builder {

        int ephemeral1hInputTokens;
        int ephemeral5mInputTokens;

        public Builder ephemeral1hInputTokens(int ephemeral1hInputTokens) {
            this.ephemeral1hInputTokens = ephemeral1hInputTokens;
            return this;
        }

        public Builder ephemeral5mInputTokens(int ephemeral5mInputTokens) {
            this.ephemeral5mInputTokens = ephemeral5mInputTokens;
            return this;
        }

        public CacheCreation build() {
            return new CacheCreation(this);
        }
    }
}
