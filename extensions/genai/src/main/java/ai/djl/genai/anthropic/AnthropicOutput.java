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

/** The Anthropic style output. */
public class AnthropicOutput {

    private String id;
    private String type;
    private String role;
    private List<Content> content;
    private String model;

    @SerializedName("stop_reason")
    private String stopReason;

    @SerializedName("stop_sequence")
    private String stopSequence;

    private Usage usage;
    private Container container;

    AnthropicOutput(Builder builder) {
        this.id = builder.id;
        this.type = builder.type;
        this.role = builder.role;
        this.content = builder.content;
        this.model = builder.model;
        this.stopReason = builder.stopReason;
        this.stopSequence = builder.stopSequence;
        this.usage = builder.usage;
        this.container = builder.container;
    }

    /**
     * Returns the model.
     *
     * @return the model
     */
    public String getId() {
        return id;
    }

    /**
     * Returns the model.
     *
     * @return the model
     */
    public String getType() {
        return type;
    }

    /**
     * Returns the model.
     *
     * @return the model
     */
    public String getRole() {
        return role;
    }

    /**
     * Returns the model.
     *
     * @return the model
     */
    public List<Content> getContent() {
        return content;
    }

    /**
     * Returns the model.
     *
     * @return the model
     */
    public String getModel() {
        return model;
    }

    /**
     * Returns the stopReason.
     *
     * @return the stopReason
     */
    public String getStopReason() {
        return stopReason;
    }

    /**
     * Returns the stopSequence.
     *
     * @return the stopSequence
     */
    public String getStopSequence() {
        return stopSequence;
    }

    /**
     * Returns the usage.
     *
     * @return the usage
     */
    public Usage getUsage() {
        return usage;
    }

    /**
     * Returns the container.
     *
     * @return the container
     */
    public Container getContainer() {
        return container;
    }

    /**
     * Creates a builder to build a {@code AnthropicOutput}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /** The builder for {@code AnthropicOutput}. */
    public static final class Builder {

        String id;
        String type;
        String role;
        List<Content> content;
        String model;
        String stopReason;
        String stopSequence;
        Usage usage;
        Container container;

        /**
         * Sets the id.
         *
         * @param id the id
         * @return the builder
         */
        public Builder id(String id) {
            this.id = id;
            return this;
        }

        /**
         * Sets the type.
         *
         * @param type the type
         * @return the builder
         */
        public Builder type(String type) {
            this.type = type;
            return this;
        }

        /**
         * Sets the role.
         *
         * @param role the role
         * @return the builder
         */
        public Builder role(String role) {
            this.role = role;
            return this;
        }

        /**
         * Sets the content.
         *
         * @param content the content
         * @return the builder
         */
        public Builder content(List<Content> content) {
            this.content = content;
            return this;
        }

        /**
         * Sets the model.
         *
         * @param model the model
         * @return the builder
         */
        public Builder model(String model) {
            this.model = model;
            return this;
        }

        /**
         * Sets the stopReason.
         *
         * @param stopReason the stopReason
         * @return the builder
         */
        public Builder stopReason(String stopReason) {
            this.stopReason = stopReason;
            return this;
        }

        /**
         * Sets the stopSequence.
         *
         * @param stopSequence the stopSequence
         * @return the builder
         */
        public Builder stopSequence(String stopSequence) {
            this.stopSequence = stopSequence;
            return this;
        }

        /**
         * Sets the usage.
         *
         * @param usage the usage
         * @return the builder
         */
        public Builder usage(Usage usage) {
            this.usage = usage;
            return this;
        }

        /**
         * Sets the container.
         *
         * @param container the container
         * @return the builder
         */
        public Builder container(Container container) {
            this.container = container;
            return this;
        }

        /**
         * Returns the {@code AnthropicOutput} instance.
         *
         * @return the {@code AnthropicOutput} instance
         */
        public AnthropicOutput build() {
            return new AnthropicOutput(this);
        }
    }
}
