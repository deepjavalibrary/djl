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
public class Tool {

    private String type;
    private String name;
    private String description;

    @SerializedName("input_schema")
    private Object inputSchema;

    @SerializedName("cache_control")
    private CacheControl cacheControl;

    Tool(Builder builder) {
        this.type = builder.type;
        this.name = builder.name;
        this.description = builder.description;
        this.inputSchema = builder.inputSchema;
        this.cacheControl = builder.cacheControl;
    }

    public String getType() {
        return type;
    }

    public String getName() {
        return name;
    }

    public String getDescription() {
        return description;
    }

    public Object getInputSchema() {
        return inputSchema;
    }

    public CacheControl getCacheControl() {
        return cacheControl;
    }

    /**
     * Creates a builder to build a {@code Tool}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /** The builder for {@code Tool}. */
    public static final class Builder {

        String type;
        String name;
        String description;
        Object inputSchema;
        CacheControl cacheControl;

        public Builder type(String type) {
            this.type = type;
            return this;
        }

        public Builder name(String name) {
            this.name = name;
            return this;
        }

        public Builder description(String description) {
            this.description = description;
            return this;
        }

        public Builder inputSchema(Object inputSchema) {
            this.inputSchema = inputSchema;
            return this;
        }

        public Builder cacheControl(CacheControl cacheControl) {
            this.cacheControl = cacheControl;
            return this;
        }

        public Tool build() {
            return new Tool(this);
        }
    }
}
