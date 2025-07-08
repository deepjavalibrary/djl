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

import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/** A data class represents text generation schema. */
public class GenerationConfig {

    private Map<String, Object> parameters;

    GenerationConfig(Map<String, Object> parameters) {
        this.parameters = parameters;
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
     * Creates a builder to build a {@code GenerationConfig}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /** The builder for {@code GenerationConfig}. */
    public static final class Builder {

        private Map<String, Object> parameters = new ConcurrentHashMap<>();

        /**
         * Sets if do_sample.
         *
         * @param doSample if do_sample
         * @return the builder
         */
        public Builder doSample(Boolean doSample) {
            parameters.put("do_sample", doSample);
            return this;
        }

        /**
         * Sets the seed.
         *
         * @param seed the seed
         * @return the builder
         */
        public Builder seed(Integer seed) {
            parameters.put("seed", seed);
            return this;
        }

        /**
         * Sets the temperature.
         *
         * @param temperature the temperature
         * @return the builder
         */
        public Builder temperature(Float temperature) {
            parameters.put("temperature", temperature);
            return this;
        }

        /**
         * Sets the repetition penalty.
         *
         * @param repetitionPenalty the repetition penalty
         * @return the builder
         */
        public Builder repetitionPenalty(Float repetitionPenalty) {
            parameters.put("repetition_penalty", repetitionPenalty);
            return this;
        }

        /**
         * Sets the top_k.
         *
         * @param topK the top_k
         * @return the builder
         */
        public Builder topK(Integer topK) {
            parameters.put("top_k", topK);
            return this;
        }

        /**
         * Sets the top_p.
         *
         * @param topP the top_p
         * @return the builder
         */
        public Builder topP(Float topP) {
            parameters.put("top_p", topP);
            return this;
        }

        /**
         * Sets the max new tokens.
         *
         * @param maxNewTokens the max new tokens
         * @return the builder
         */
        public Builder maxNewTokens(Integer maxNewTokens) {
            parameters.put("max_new_tokens", maxNewTokens);
            return this;
        }

        /**
         * Sets if return the details.
         *
         * @param details if return the details
         * @return the builder
         */
        public Builder details(Boolean details) {
            parameters.put("details", details);
            return this;
        }

        /**
         * Sets if return full text.
         *
         * @param returnFullText if return full text
         * @return the builder
         */
        public Builder returnFullText(Boolean returnFullText) {
            parameters.put("return_full_text", returnFullText);
            return this;
        }

        /**
         * Sets the stop sequences.
         *
         * @param stopSequences the stop sequences
         * @return the builder
         */
        public Builder stopSequences(List<String> stopSequences) {
            parameters.put("stop_sequences", stopSequences);
            return this;
        }

        /**
         * Sets if return the decoder input details.
         *
         * @param decoderInputDetails if return the decoder input details
         * @return the builder
         */
        public Builder decoderInputDetails(Boolean decoderInputDetails) {
            parameters.put("decoder_input_details", decoderInputDetails);
            return this;
        }

        /**
         * Sets the custom text generation parameter.
         *
         * @param key the parameter key
         * @param value the parameter value
         * @return the builder
         */
        public Builder addParameter(String key, Object value) {
            parameters.put(key, value);
            return this;
        }

        /**
         * Returns a new {@code GenerationConfig} instance.
         *
         * @return a new {@code GenerationConfig} instance
         */
        public GenerationConfig build() {
            return new GenerationConfig(parameters);
        }
    }
}
