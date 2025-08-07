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
public class McpServer {

    private String type;
    private String name;
    private String url;

    @SerializedName("authorization_token")
    private String authorizationToken;

    @SerializedName("tool_configuration")
    private ToolConfiguration toolConfiguration;

    McpServer(Builder builder) {
        this.type = builder.type;
        this.name = builder.name;
        this.url = builder.url;
        this.authorizationToken = builder.authorizationToken;
        this.toolConfiguration = builder.toolConfiguration;
    }

    public String getType() {
        return type;
    }

    public String getName() {
        return name;
    }

    public String getUrl() {
        return url;
    }

    public String getAuthorizationToken() {
        return authorizationToken;
    }

    public ToolConfiguration getToolConfiguration() {
        return toolConfiguration;
    }

    /**
     * Creates a builder to build a {@code McpServer}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /** The builder for {@code McpServer}. */
    public static final class Builder {

        String type;
        String name;
        String url;
        String authorizationToken;
        ToolConfiguration toolConfiguration;

        public Builder type(String type) {
            this.type = type;
            return this;
        }

        public Builder name(String name) {
            this.name = name;
            return this;
        }

        public Builder url(String url) {
            this.url = url;
            return this;
        }

        public Builder authorizationToken(String authorizationToken) {
            this.authorizationToken = authorizationToken;
            return this;
        }

        public Builder toolConfiguration(ToolConfiguration toolConfiguration) {
            this.toolConfiguration = toolConfiguration;
            return this;
        }

        public McpServer build() {
            return new McpServer(this);
        }
    }
}
