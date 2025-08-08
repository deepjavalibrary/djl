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

import java.util.List;

/** A data class represents Anthropic schema. */
@SuppressWarnings("MissingJavadocMethod")
public class ToolConfiguration {

    @SerializedName("allowed_tools")
    private List<String> allowedTools;

    private Boolean enabled;

    ToolConfiguration(Builder builder) {
        this.allowedTools = builder.allowedTools;
        this.enabled = builder.enabled;
    }

    public List<String> getAllowedTools() {
        return allowedTools;
    }

    public Boolean getEnabled() {
        return enabled;
    }

    /**
     * Creates a builder to build a {@code ToolConfiguration}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /** The builder for {@code ToolConfiguration}. */
    public static final class Builder {

        List<String> allowedTools;
        Boolean enabled;

        public Builder allowedTools(List<String> allowedTools) {
            this.allowedTools = allowedTools;
            return this;
        }

        public Builder enabled(Boolean enabled) {
            this.enabled = enabled;
            return this;
        }

        public ToolConfiguration build() {
            return new ToolConfiguration(this);
        }
    }
}
