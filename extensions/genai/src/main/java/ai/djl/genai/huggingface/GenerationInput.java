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

import java.util.Map;

/** A data class represents text generation schema. */
public class GenerationInput {

    private Object inputs;
    private Map<String, Object> parameters;
    private Boolean stream;

    GenerationInput(Builder builder) {
        this.inputs = builder.inputs;
        this.parameters = builder.parameters;
        this.stream = builder.stream;
    }

    /**
     * Returns the inputs.
     *
     * @return the inputs
     */
    public Object getInputs() {
        return inputs;
    }

    /**
     * Returns the text generation parameters.
     *
     * @return the text generation parameters
     */
    public Map<String, Object> getParameters() {
        return parameters;
    }

    /**
     * Returns if streaming response.
     *
     * @return if streaming response
     */
    public Boolean getStream() {
        return stream;
    }

    /**
     * Creates a builder to build a {@code GenerationInput}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Creates a builder with the specified input text.
     *
     * @param text the input text
     * @return a new builder
     */
    public static Builder text(String text) {
        return builder().input(text);
    }

    /**
     * Creates a builder with the specified text content.
     *
     * @param inputs the inputs
     * @return a new builder
     */
    public static Builder text(String[] inputs) {
        return builder().inputs(inputs);
    }

    /** The builder for {@code ChatInput}. */
    public static final class Builder {

        Object inputs;
        Map<String, Object> parameters;
        Boolean stream;

        /**
         * Sets the input text.
         *
         * @param text the input text
         * @return the builder
         */
        public Builder input(String text) {
            this.inputs = text;
            return this;
        }

        /**
         * Sets the input messages.
         *
         * @param inputs the input messages
         * @return the builder
         */
        public Builder inputs(String... inputs) {
            this.inputs = inputs;
            return this;
        }

        /**
         * Sets the input messages.
         *
         * @param inputs the input messages
         * @return the builder
         */
        public Builder inputs(Object inputs) {
            this.inputs = inputs;
            return this;
        }

        /**
         * Sets the generation parameters.
         *
         * @param config generation parameters
         * @return the builder
         */
        public Builder config(GenerationConfig config) {
            this.parameters = config.getParameters();
            return this;
        }

        /**
         * Sets the generation parameters.
         *
         * @param config generation parameters
         * @return the builder
         */
        public Builder config(GenerationConfig.Builder config) {
            return config(config.build());
        }

        /**
         * Sets if return response in stream.
         *
         * @param stream if return response in stream
         * @return the builder
         */
        public Builder stream(Boolean stream) {
            this.stream = stream;
            return this;
        }

        /**
         * Builds the {@code ChatInput} instance.
         *
         * @return the {@code ChatInput} instance
         */
        public GenerationInput build() {
            return new GenerationInput(this);
        }
    }
}
