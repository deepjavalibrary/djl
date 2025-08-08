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
public class ToolChoice {

    private String type;
    private String name;

    @SerializedName("disable_parallel_tool_use")
    private boolean disableParallelToolUse;

    ToolChoice(Builder builder) {
        this.type = builder.type;
        this.name = builder.name;
        this.disableParallelToolUse = builder.disableParallelToolUse;
    }

    public String getType() {
        return type;
    }

    public String getName() {
        return name;
    }

    public boolean isDisableParallelToolUse() {
        return disableParallelToolUse;
    }

    /**
     * Creates a builder to build a {@code ToolChoice}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /** The builder for {@code ToolChoice}. */
    public static final class Builder {

        String type;
        String name;
        boolean disableParallelToolUse;

        public Builder type(String type) {
            this.type = type;
            return this;
        }

        public Builder name(String name) {
            this.name = name;
            return this;
        }

        public Builder disableParallelToolUse(boolean disableParallelToolUse) {
            this.disableParallelToolUse = disableParallelToolUse;
            return this;
        }

        public ToolChoice build() {
            return new ToolChoice(this);
        }
    }
}
