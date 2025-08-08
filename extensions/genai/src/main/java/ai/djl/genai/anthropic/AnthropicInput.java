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

import java.util.ArrayList;
import java.util.List;

/** The Anthropic style input. */
public class AnthropicInput {

    @SerializedName("anthropic_version")
    private String anthropicVersion;

    private String model;

    @SerializedName("max_tokens")
    private Integer maxTokens;

    private List<Message> messages;
    private String container;

    @SerializedName("mcp_servers")
    private List<McpServer> mcpServers;

    private Metadata metadata;

    @SerializedName("service_tier")
    private String serviceTier;

    @SerializedName("stop_sequences")
    private List<String> stopSequences;

    private Boolean stream;

    @SerializedName("system")
    private String systemInstructions;

    private Float temperature;
    private Thinking thinking;

    @SerializedName("tool_choice")
    private ToolChoice toolChoice;

    private List<Tool> tools;

    @SerializedName("top_k")
    private Integer topK;

    @SerializedName("top_p")
    private Float topP;

    AnthropicInput(Builder builder) {
        this.anthropicVersion = builder.anthropicVersion;
        this.model = builder.model;
        this.maxTokens = builder.maxTokens;
        this.messages = builder.messages;
        this.container = builder.container;
        this.mcpServers = builder.mcpServers;
        this.metadata = builder.metadata;
        this.serviceTier = builder.serviceTier;
        this.stopSequences = builder.stopSequences;
        this.stream = builder.stream;
        this.systemInstructions = builder.systemInstructions;
        this.temperature = builder.temperature;
        this.thinking = builder.thinking;
        this.toolChoice = builder.toolChoice;
        this.tools = builder.tools;
        this.topK = builder.topK;
        this.topP = builder.topP;
    }

    /**
     * Returns the Anthropic version.
     *
     * @return the Anthropic version
     */
    public String getAnthropicVersion() {
        return anthropicVersion;
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
     * Returns the maxTokens.
     *
     * @return the maxTokens
     */
    public Integer getMaxTokens() {
        return maxTokens;
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
     * Returns the container.
     *
     * @return the container
     */
    public String getContainer() {
        return container;
    }

    /**
     * Returns the mcpServers.
     *
     * @return the mcpServers
     */
    public List<McpServer> getMcpServers() {
        return mcpServers;
    }

    /**
     * Returns the metadata.
     *
     * @return the metadata
     */
    public Metadata getMetadata() {
        return metadata;
    }

    /**
     * Returns the serviceTier.
     *
     * @return the serviceTier
     */
    public String getServiceTier() {
        return serviceTier;
    }

    /**
     * Returns the stopSequences.
     *
     * @return the stopSequences
     */
    public List<String> getStopSequences() {
        return stopSequences;
    }

    /**
     * Returns the stream.
     *
     * @return the stream
     */
    public Boolean getStream() {
        return stream;
    }

    /**
     * Returns the systemInstructions.
     *
     * @return the systemInstructions
     */
    public String getSystemInstructions() {
        return systemInstructions;
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
     * Returns the thinking.
     *
     * @return the thinking
     */
    public Thinking getThinking() {
        return thinking;
    }

    /**
     * Returns the toolChoice.
     *
     * @return the toolChoice
     */
    public ToolChoice getToolChoice() {
        return toolChoice;
    }

    /**
     * Returns the tools.
     *
     * @return the tools
     */
    public List<Tool> getTools() {
        return tools;
    }

    /**
     * Returns the topK.
     *
     * @return the topK
     */
    public Integer getTopK() {
        return topK;
    }

    /**
     * Returns the topP.
     *
     * @return the topP
     */
    public Float getTopP() {
        return topP;
    }

    /**
     * Creates a builder to build a {@code AnthropicInput}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /** The builder for {@code AnthropicInput}. */
    public static final class Builder {
        String anthropicVersion;
        String model;
        Integer maxTokens = 512;
        List<Message> messages = new ArrayList<>();
        String container;
        List<McpServer> mcpServers;
        Metadata metadata;
        String serviceTier;
        List<String> stopSequences;
        Boolean stream;
        String systemInstructions;
        Float temperature;
        Thinking thinking;
        ToolChoice toolChoice;
        List<Tool> tools;
        Integer topK;
        Float topP;

        /**
         * Sets the anthropicVersion.
         *
         * @param anthropicVersion the anthropicVersion
         * @return the builder
         */
        public Builder anthropicVersion(String anthropicVersion) {
            this.anthropicVersion = anthropicVersion;
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
         * Sets the maxTokens.
         *
         * @param maxTokens the maxTokens
         * @return the builder
         */
        public Builder maxTokens(Integer maxTokens) {
            this.maxTokens = maxTokens;
            return this;
        }

        /**
         * Sets the messages.
         *
         * @param messages the messages
         * @return the builder
         */
        public Builder messages(List<Message> messages) {
            this.messages = messages;
            return this;
        }

        /**
         * Adds the message.
         *
         * @param message the message
         * @return the builder
         */
        public Builder addMessage(Message.Builder message) {
            return addMessage(message.build());
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
         * Sets the container.
         *
         * @param container the container
         * @return the builder
         */
        public Builder container(String container) {
            this.container = container;
            return this;
        }

        /**
         * Sets the mcpServers.
         *
         * @param mcpServers the mcpServers
         * @return the builder
         */
        public Builder mcpServers(List<McpServer> mcpServers) {
            this.mcpServers = mcpServers;
            return this;
        }

        /**
         * Sets the metadata.
         *
         * @param metadata the metadata
         * @return the builder
         */
        public Builder metadata(Metadata metadata) {
            this.metadata = metadata;
            return this;
        }

        /**
         * Sets the serviceTier.
         *
         * @param serviceTier the serviceTier
         * @return the builder
         */
        public Builder serviceTier(String serviceTier) {
            this.serviceTier = serviceTier;
            return this;
        }

        /**
         * Sets the stopSequences.
         *
         * @param stopSequences the stopSequences
         * @return the builder
         */
        public Builder stopSequences(List<String> stopSequences) {
            this.stopSequences = stopSequences;
            return this;
        }

        /**
         * Sets the stream.
         *
         * @param stream the stream
         * @return the builder
         */
        public Builder stream(Boolean stream) {
            this.stream = stream;
            return this;
        }

        /**
         * Sets the system instructions.
         *
         * @param systemInstructions the system instructions
         * @return the builder
         */
        public Builder systemInstructions(String systemInstructions) {
            this.systemInstructions = systemInstructions;
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
         * Sets the thinking.
         *
         * @param thinking the thinking
         * @return the builder
         */
        public Builder thinking(Thinking thinking) {
            this.thinking = thinking;
            return this;
        }

        /**
         * Sets the tool choice.
         *
         * @param toolChoice the tool choice
         * @return the builder
         */
        public Builder toolChoice(ToolChoice toolChoice) {
            this.toolChoice = toolChoice;
            return this;
        }

        /**
         * Sets the tools.
         *
         * @param tools the tools
         * @return the builder
         */
        public Builder tools(List<Tool> tools) {
            this.tools = tools;
            return this;
        }

        /**
         * Adds the tool.
         *
         * @param tool the tool
         * @return the builder
         */
        public Builder addTool(Tool tool) {
            if (tools == null) {
                tools = new ArrayList<>();
            }
            tools.add(tool);
            return this;
        }

        /**
         * Sets the topK.
         *
         * @param topK the topK
         * @return the builder
         */
        public Builder topK(Integer topK) {
            this.topK = topK;
            return this;
        }

        /**
         * Sets the topP.
         *
         * @param topP the topP
         * @return the builder
         */
        public Builder topP(Float topP) {
            this.topP = topP;
            return this;
        }

        /**
         * Returns the {@code AnthropicInput} instance.
         *
         * @return the {@code AnthropicInput} instance
         */
        public AnthropicInput build() {
            return new AnthropicInput(this);
        }
    }
}
