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

import com.google.gson.annotations.SerializedName;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/** The chat completion style input. */
public class ChatInput {

    private String model;
    private List<Message> messages;

    @SerializedName("frequency_penalty")
    private Float frequencyPenalty;

    @SerializedName("logit_bias")
    Map<String, Double> logitBias;

    private Boolean logprobs;

    @SerializedName("topLogprobs")
    private Integer topLogprobs;

    @SerializedName("max_completion_tokens")
    private Integer maxCompletionTokens;

    private Integer n;

    @SerializedName("presence_penalty")
    private Float presencePenalty;

    private Integer seed;
    private List<String> stop;
    private Boolean stream;
    private Float temperature;

    @SerializedName("top_p")
    private Float topP;

    private String user;

    @SerializedName("ignore_eos")
    private Boolean ignoreEos;

    private List<Tool> tools;

    @SerializedName("tool_choice")
    private Object toolChoice;

    ChatInput(Builder builder) {
        model = builder.model;
        messages = builder.messages;
        frequencyPenalty = builder.frequencyPenalty;
        logitBias = builder.logitBias;
        logprobs = builder.logprobs;
        topLogprobs = builder.topLogprobs;
        maxCompletionTokens = builder.maxCompletionTokens;
        n = builder.n;
        presencePenalty = builder.presencePenalty;
        seed = builder.seed;
        stop = builder.stop;
        stream = builder.stream;
        temperature = builder.temperature;
        topP = builder.topP;
        user = builder.user;
        ignoreEos = builder.ignoreEos;
        tools = builder.tools;
        toolChoice = builder.toolChoice;
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
     * Returns the messages.
     *
     * @return the messages
     */
    public List<Message> getMessages() {
        return messages;
    }

    /**
     * Returns the frequency penalty.
     *
     * @return the frequency penalty
     */
    public Float getFrequencyPenalty() {
        return frequencyPenalty;
    }

    /**
     * Returns the logit bias.
     *
     * @return the logit bias
     */
    public Map<String, Double> getLogitBias() {
        return logitBias;
    }

    /**
     * Returns the logprobs.
     *
     * @return the logprobs
     */
    public Boolean getLogprobs() {
        return logprobs;
    }

    /**
     * Returns the top logprobs.
     *
     * @return the top logprobs
     */
    public Integer getTopLogprobs() {
        return topLogprobs;
    }

    /**
     * Returns the max tokens.
     *
     * @return the max tokens
     */
    public Integer getMaxCompletionTokens() {
        return maxCompletionTokens;
    }

    /**
     * Returns the N.
     *
     * @return the n
     */
    public Integer getN() {
        return n;
    }

    /**
     * Returns the presence penalty.
     *
     * @return the presence penalty
     */
    public Float getPresencePenalty() {
        return presencePenalty;
    }

    /**
     * Returns the seed.
     *
     * @return the seed
     */
    public Integer getSeed() {
        return seed;
    }

    /**
     * Returns the stop char sequences.
     *
     * @return the stop char sequences
     */
    public List<String> getStop() {
        return stop;
    }

    /**
     * Returns true if {@code stream} is enabled.
     *
     * @return true if {@code stream} is enabled
     */
    public Boolean getStream() {
        return stream;
    }

    /**
     * Returns the temperature.
     *
     * @return the temperature
     */
    public Float getTemperature() {
        return temperature;
    }

    /**
     * Returns the top p value.
     *
     * @return the top p value
     */
    public Float getTopP() {
        return topP;
    }

    /**
     * Returns the user.
     *
     * @return the user
     */
    public String getUser() {
        return user;
    }

    /**
     * Returns if ignore eos.
     *
     * @return if ignore eos
     */
    public Boolean getIgnoreEos() {
        return ignoreEos;
    }

    /**
     * Returns the {@link Tool}s.
     *
     * @return the {@code Tool}s
     */
    public List<Tool> getTools() {
        return tools;
    }

    /**
     * Returns the tool choice.
     *
     * @return the tool choice
     */
    public Object getToolChoice() {
        return toolChoice;
    }

    /**
     * Creates a builder to build a {@code ChatInput}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Creates a builder with the specified model.
     *
     * @param model the model
     * @return a new builder
     */
    public static Builder model(String model) {
        return builder().model(model);
    }

    /**
     * Creates a builder with the specified text content.
     *
     * @param text the text
     * @return a new builder
     */
    public static Builder text(String text) {
        return builder().addMessage(Message.fromText(text));
    }

    /**
     * Creates a builder with the specified file uri.
     *
     * @param imageUrl the file uri
     * @return a new builder
     */
    public static Builder image(String imageUrl) {
        return builder().addImage(imageUrl);
    }

    /**
     * Creates a builder with the specified file data.
     *
     * @param id the file id
     * @param data the file data
     * @param fileName the file name
     * @return a new builder
     */
    public static Builder file(String id, byte[] data, String fileName) {
        return builder().addFile(id, data, fileName);
    }

    /** The builder for {@code ChatInput}. */
    public static final class Builder {
        String model;
        List<Message> messages = new ArrayList<>();
        Float frequencyPenalty;
        Map<String, Double> logitBias;
        Boolean logprobs;
        Integer topLogprobs;
        Integer maxCompletionTokens;
        Integer n;
        Float presencePenalty;
        Integer seed;
        List<String> stop;
        Boolean stream;
        Float temperature;
        Float topP;
        String user;
        Boolean ignoreEos;
        List<Tool> tools;
        Object toolChoice;

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
         * Sets the messages.
         *
         * @param messages the messages
         * @return the builder
         */
        public Builder messages(List<Message> messages) {
            this.messages.clear();
            this.messages.addAll(messages);
            return this;
        }

        /**
         * Adds the message.
         *
         * @param message the message
         * @return the builder
         */
        public Builder addMessage(Message message) {
            this.messages.add(message);
            return this;
        }

        /**
         * Adds the text content.
         *
         * @param text the text
         * @return the builder
         */
        public Builder addText(String text) {
            this.messages.add(Message.fromText(text));
            return this;
        }

        /**
         * Adds the image content.
         *
         * @param imageUrl the image url
         * @return the builder
         */
        public Builder addImage(String imageUrl) {
            this.messages.add(Message.fromImage(imageUrl));
            return this;
        }

        /**
         * Adds the file content.
         *
         * @param data the file data
         * @param fileName the mime type
         * @return the builder
         */
        public Builder addFile(String id, byte[] data, String fileName) {
            this.messages.add(Message.fromFile(id, data, fileName));
            return this;
        }

        /**
         * Sets the frequency penalty.
         *
         * @param frequencyPenalty the frequency penalty
         * @return the builder
         */
        public Builder frequencyPenalty(Float frequencyPenalty) {
            this.frequencyPenalty = frequencyPenalty;
            return this;
        }

        /**
         * Sets the logit bias.
         *
         * @param logitBias the logit bias
         * @return the builder
         */
        public Builder logitBias(Map<String, Double> logitBias) {
            this.logitBias = logitBias;
            return this;
        }

        /**
         * Sets the logprobs.
         *
         * @param logprobs the logprobs
         * @return the builder
         */
        public Builder logprobs(Boolean logprobs) {
            this.logprobs = logprobs;
            return this;
        }

        /**
         * Sets the top logprobs.
         *
         * @param topLogprobs the top logprobs
         * @return the builder
         */
        public Builder topLogprobs(Integer topLogprobs) {
            this.topLogprobs = topLogprobs;
            return this;
        }

        /**
         * Sets the max tokens.
         *
         * @param maxCompletionTokens the max tokens
         * @return the builder
         */
        public Builder maxCompletionTokens(Integer maxCompletionTokens) {
            this.maxCompletionTokens = maxCompletionTokens;
            return this;
        }

        /**
         * Sets the N.
         *
         * @param n the N
         * @return the builder
         */
        public Builder n(Integer n) {
            this.n = n;
            return this;
        }

        /**
         * Sets the presence penalty.
         *
         * @param presencePenalty the presence penalty
         * @return the builder
         */
        public Builder presencePenalty(Float presencePenalty) {
            this.presencePenalty = presencePenalty;
            return this;
        }

        /**
         * Sets the seed.
         *
         * @param seed the seed
         * @return the builder
         */
        public Builder seed(Integer seed) {
            this.seed = seed;
            return this;
        }

        /**
         * Sets the stop sequences.
         *
         * @param stop the stop sequences
         * @return the builder
         */
        public Builder stop(List<String> stop) {
            this.stop = stop;
            return this;
        }

        /**
         * Sets stream mode.
         *
         * @param stream the stream mode
         * @return the builder
         */
        public Builder stream(Boolean stream) {
            this.stream = stream;
            return this;
        }

        /**
         * Sets the temperature.
         *
         * @param temperature the temperature
         * @return the builder
         */
        public Builder temperature(Float temperature) {
            this.temperature = temperature;
            return this;
        }

        /**
         * Sets the topP.
         *
         * @param topP the top p
         * @return the builder
         */
        public Builder topP(Float topP) {
            this.topP = topP;
            return this;
        }

        /**
         * Sets the user.
         *
         * @param user the user
         * @return the builder
         */
        public Builder user(String user) {
            this.user = user;
            return this;
        }

        /**
         * Sets if ignore eos.
         *
         * @param ignoreEos if ignore eos
         * @return the builder
         */
        public Builder ignoreEos(Boolean ignoreEos) {
            this.ignoreEos = ignoreEos;
            return this;
        }

        /**
         * Sets the tools.
         *
         * @param tools the tools
         * @return the tools
         */
        public Builder tools(List<Tool> tools) {
            this.tools = tools;
            return this;
        }

        /**
         * Sets the tool choice mode.
         *
         * @param toolChoice the tool choice mode
         * @return the builder
         */
        public Builder toolChoice(String toolChoice) {
            this.toolChoice = toolChoice;
            return this;
        }

        /**
         * Sets the tool choice.
         *
         * @param toolChoice the tool choice
         * @return the builder
         */
        public Builder toolChoice(Tool toolChoice) {
            this.toolChoice = toolChoice;
            return this;
        }

        /**
         * Builds the {@code ChatInput} instance.
         *
         * @return the {@code ChatInput} instance
         */
        public ChatInput build() {
            if (model == null) {
                throw new IllegalArgumentException("model is not specified");
            }
            return new ChatInput(this);
        }
    }
}
