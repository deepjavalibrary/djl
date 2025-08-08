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
public class Usage {

    @SerializedName("cache_creation")
    private CacheCreation cacheCreation;

    @SerializedName("cache_creation_input_tokens")
    private int cacheCreationInputTokens;

    @SerializedName("cache_read_input_tokens")
    private int cacheReadInputTokens;

    @SerializedName("input_tokens")
    private int inputTokens;

    @SerializedName("output_tokens")
    private int outputTokens;

    @SerializedName("server_tool_use")
    private ServerToolUse serverToolUse;

    @SerializedName("service_tier")
    private String serviceTier;

    Usage(Builder builder) {
        this.cacheCreation = builder.cacheCreation;
        this.cacheCreationInputTokens = builder.cacheCreationInputTokens;
        this.cacheReadInputTokens = builder.cacheReadInputTokens;
        this.inputTokens = builder.inputTokens;
        this.outputTokens = builder.outputTokens;
        this.serverToolUse = builder.serverToolUse;
        this.serviceTier = builder.serviceTier;
    }

    public CacheCreation getCacheCreation() {
        return cacheCreation;
    }

    public int getCacheCreationInputTokens() {
        return cacheCreationInputTokens;
    }

    public int getCacheReadInputTokens() {
        return cacheReadInputTokens;
    }

    public int getInputTokens() {
        return inputTokens;
    }

    public int getOutputTokens() {
        return outputTokens;
    }

    public ServerToolUse getServerToolUse() {
        return serverToolUse;
    }

    public String getServiceTier() {
        return serviceTier;
    }

    /**
     * Creates a builder to build a {@code Usage}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /** The builder for {@code Usage}. */
    public static final class Builder {

        CacheCreation cacheCreation;
        int cacheCreationInputTokens;
        int cacheReadInputTokens;
        int inputTokens;
        int outputTokens;
        ServerToolUse serverToolUse;
        String serviceTier;

        public Builder cacheCreation(CacheCreation cacheCreation) {
            this.cacheCreation = cacheCreation;
            return this;
        }

        public Builder cacheCreationInputTokens(int cacheCreationInputTokens) {
            this.cacheCreationInputTokens = cacheCreationInputTokens;
            return this;
        }

        public Builder cacheReadInputTokens(int cacheReadInputTokens) {
            this.cacheReadInputTokens = cacheReadInputTokens;
            return this;
        }

        public Builder inputTokens(int inputTokens) {
            this.inputTokens = inputTokens;
            return this;
        }

        public Builder outputTokens(int outputTokens) {
            this.outputTokens = outputTokens;
            return this;
        }

        public Builder serverToolUse(ServerToolUse serverToolUse) {
            this.serverToolUse = serverToolUse;
            return this;
        }

        public Builder serviceTier(String serviceTier) {
            this.serviceTier = serviceTier;
            return this;
        }

        public Usage build() {
            return new Usage(this);
        }
    }
}
