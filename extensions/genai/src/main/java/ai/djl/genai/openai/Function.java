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
package ai.djl.genai.openai;

/** A data class represents chat completion schema. */
@SuppressWarnings("MissingJavadocMethod")
public class Function {

    private String name;
    private String description;
    private Object parameters;

    public Function(Builder builder) {
        name = builder.name;
        description = builder.description;
        parameters = builder.parameters;
    }

    public String getName() {
        return name;
    }

    public String getDescription() {
        return description;
    }

    public Object getParameters() {
        return parameters;
    }

    /**
     * Creates a builder to build a {@code Function}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /** The builder for {@code Function}. */
    public static final class Builder {

        String name;
        String description;
        Object parameters;

        public Builder name(String name) {
            this.name = name;
            return this;
        }

        public Builder description(String description) {
            this.description = description;
            return this;
        }

        public Builder parameters(String parameters) {
            this.parameters = parameters;
            return this;
        }

        public Function build() {
            return new Function(this);
        }
    }
}
