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

import ai.djl.genai.anthropic.AnthropicInput;
import ai.djl.genai.anthropic.Source;
import ai.djl.genai.anthropic.ToolChoice;
import ai.djl.genai.gemini.GeminiInput;
import ai.djl.genai.gemini.types.Blob;
import ai.djl.genai.gemini.types.FunctionDeclaration;
import ai.djl.genai.gemini.types.GenerationConfig;
import ai.djl.genai.gemini.types.Part;
import ai.djl.genai.gemini.types.Schema;
import ai.djl.genai.gemini.types.ThinkingConfig;
import ai.djl.util.JsonSerializable;
import ai.djl.util.JsonUtils;

import com.google.gson.Gson;
import com.google.gson.JsonElement;
import com.google.gson.annotations.SerializedName;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/** The chat completion style input. */
@SuppressWarnings("serial")
public class ChatInput implements JsonSerializable {

    private static final long serialVersionUID = 1L;

    private static final Logger logger = LoggerFactory.getLogger(ChatInput.class);

    private static final Pattern URL_PATTERN = Pattern.compile("data:([\\w/]+);base64,(.+)");
    static final Gson GSON = new Gson();

    private transient Type inputType;
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

    @SerializedName("reasoning_effort")
    private String reasoningEffort;

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

    @SerializedName("extra_body")
    private Object extraBody;

    ChatInput(Builder builder) {
        inputType = builder.inputType;
        model = builder.model;
        messages = builder.messages;
        frequencyPenalty = builder.frequencyPenalty;
        logitBias = builder.logitBias;
        logprobs = builder.logprobs;
        topLogprobs = builder.topLogprobs;
        maxCompletionTokens = builder.maxCompletionTokens;
        n = builder.n;
        presencePenalty = builder.presencePenalty;
        reasoningEffort = builder.reasoningEffort;
        seed = builder.seed;
        stop = builder.stop;
        stream = builder.stream;
        temperature = builder.temperature;
        topP = builder.topP;
        user = builder.user;
        ignoreEos = builder.ignoreEos;
        tools = builder.tools;
        toolChoice = builder.toolChoice;
        extraBody = builder.extraBody;
    }

    /** {@inheritDoc} */
    @Override
    public JsonElement serialize() {
        if (inputType == Type.GEMINI) {
            return JsonUtils.GSON.toJsonTree(toGemini());
        } else if (inputType == Type.ANTHROPIC || inputType == Type.ANTHROPIC_VERTEX) {
            return JsonUtils.GSON.toJsonTree(toAnthropic());
        }
        return GSON.toJsonTree(this);
    }

    @SuppressWarnings("unchecked")
    private GeminiInput toGemini() {
        GeminiInput.Builder builder = GeminiInput.builder();
        GenerationConfig.Builder config = GenerationConfig.builder();
        for (Message message : messages) {
            String role = message.getRole();
            Object obj = message.getContent();
            ai.djl.genai.gemini.types.Content.Builder cb =
                    ai.djl.genai.gemini.types.Content.builder();
            if (obj instanceof String) {
                cb.addPart(Part.text((String) obj));
            } else {
                for (Content content : (List<Content>) obj) {
                    String type = content.getType();
                    if ("image_url".equals(type)) {
                        Content.ImageContent ic = content.getImageUrl();
                        String url = ic.getUrl();
                        Matcher m = URL_PATTERN.matcher(url);
                        if (m.matches()) {
                            Blob blob =
                                    Blob.builder().data(m.group(2)).mimeType(m.group(1)).build();
                            cb.addPart(Part.builder().inlineData(blob));
                        } else {
                            Blob blob = Blob.builder().data(url).build();
                            cb.addPart(Part.builder().inlineData(blob));
                        }
                    } else if ("file".equals(type)) {
                        Content.FileContent fc = content.getFile();
                        cb.addPart(Part.fileData(fc.getFileData(), null));
                    } else if ("text".equals(type)) {
                        cb.addPart(Part.text(content.getText()));
                    } else {
                        throw new IllegalArgumentException("Unsupported type: " + type);
                    }
                }
            }

            if ("system".equals(role)) {
                config.systemInstruction(cb.build().getParts().get(0).getText());
            } else {
                builder.addContent(cb.role(role));
            }
        }
        if (tools != null && !tools.isEmpty()) {
            for (Tool tool : tools) {
                if (!"function".equals(tool.getType())) {
                    logger.warn("Unsupported tool type: {}", tool.getType());
                    continue;
                }
                Function function = tool.getFunction();
                Object param = function.getParameters();
                Map<String, Object> parameters = (Map<String, Object>) param;
                Map<String, Map<String, String>> properties =
                        (Map<String, Map<String, String>>) parameters.get("properties");
                List<String> required = (List<String>) parameters.get("required");
                String returnType = ((String) parameters.get("type")).toUpperCase(Locale.ROOT);

                Map<String, Schema> prop = new LinkedHashMap<>(); // NOPMD
                for (Map.Entry<String, Map<String, String>> entry : properties.entrySet()) {
                    String t = entry.getValue().get("type").toUpperCase(Locale.ROOT);
                    Schema schema =
                            Schema.builder()
                                    .type(ai.djl.genai.gemini.types.Type.valueOf(t))
                                    .build();
                    prop.put(entry.getKey(), schema);
                }
                Schema sc =
                        Schema.builder()
                                .type(ai.djl.genai.gemini.types.Type.valueOf(returnType))
                                .required(required)
                                .properties(prop)
                                .build();

                FunctionDeclaration fd =
                        FunctionDeclaration.builder()
                                .name(function.getName())
                                .description(function.getDescription())
                                .parameters(sc)
                                .build();

                ai.djl.genai.gemini.types.Tool t =
                        ai.djl.genai.gemini.types.Tool.builder().addFunctionDeclaration(fd).build();
                config.addTool(t);
            }
        }
        config.responseLogprobs(logprobs);
        config.logprobs(topLogprobs);
        config.frequencyPenalty(frequencyPenalty);
        config.presencePenalty(presencePenalty);
        config.maxOutputTokens(maxCompletionTokens);
        config.seed(seed);
        config.stopSequences(stop);
        config.candidateCount(n);
        config.topP(topP);
        config.temperature(temperature);
        if ("high".equalsIgnoreCase(reasoningEffort)) {
            config.thinkingConfig(ThinkingConfig.builder().includeThoughts(true));
        } else if ("medium".equalsIgnoreCase(reasoningEffort)) {
            config.thinkingConfig(
                    ThinkingConfig.builder().includeThoughts(true).thinkingBudget(512));
        }
        builder.generationConfig(config.build());
        return builder.build();
    }

    @SuppressWarnings("unchecked")
    private AnthropicInput toAnthropic() {
        AnthropicInput.Builder builder = AnthropicInput.builder();
        builder.model(model).stream(stream).stopSequences(stop).temperature(temperature).topP(topP);
        if (maxCompletionTokens != null) {
            builder.maxTokens(maxCompletionTokens);
        }
        for (Message message : messages) {
            String role = message.getRole();
            if ("system".equals(role)) {
                builder.systemInstructions((String) message.getContent());
                continue;
            }

            ai.djl.genai.anthropic.Message.Builder mb = ai.djl.genai.anthropic.Message.builder();
            mb.role(role);
            Object obj = message.getContent();
            if (obj instanceof String) {
                mb.text((String) obj);
            } else {
                for (Content content : (List<Content>) obj) {
                    String type = content.getType();
                    if ("image_url".equals(type)) {
                        Content.ImageContent ic = content.getImageUrl();
                        String url = ic.getUrl();
                        Matcher m = URL_PATTERN.matcher(url);
                        if (m.matches()) {
                            String mimeType = m.group(1);
                            String data = m.group(2);
                            ai.djl.genai.anthropic.Content.Builder cb =
                                    ai.djl.genai.anthropic.Content.builder();
                            cb.type("image")
                                    .source(
                                            Source.builder()
                                                    .type("base64")
                                                    .mediaType(mimeType)
                                                    .data(data));
                            mb.addContent(cb);
                        } else {
                            mb.addContent(ai.djl.genai.anthropic.Content.image(url));
                        }
                    } else if ("text".equals(type)) {
                        mb.addContent(ai.djl.genai.anthropic.Content.text(content.getText()));
                    } else {
                        throw new IllegalArgumentException("Unsupported type: " + type);
                    }
                }
            }
            builder.addMessage(mb);
        }
        if (tools != null && !tools.isEmpty()) {
            for (Tool tool : tools) {
                if (!"function".equals(tool.getType())) {
                    logger.warn("Unsupported tool type: {}", tool.getType());
                    continue;
                }
                Function function = tool.getFunction();
                Object param = function.getParameters();
                ai.djl.genai.anthropic.Tool t =
                        ai.djl.genai.anthropic.Tool.builder()
                                .name(function.getName())
                                .description(function.getDescription())
                                .inputSchema(param)
                                .build();
                builder.addTool(t);
            }
        }
        if ("auto".equals(toolChoice)) {
            builder.toolChoice(ToolChoice.builder().type("auto").build());
        }
        if (inputType == Type.ANTHROPIC_VERTEX) {
            builder.anthropicVersion("vertex-2023-10-16");
        } else if (inputType == Type.ANTHROPIC) {
            builder.anthropicVersion("2023-10-16");
        }
        return builder.build();
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
     * Returns the reasoning effort.
     *
     * @return the reasoning effort
     */
    public String getReasoningEffort() {
        return reasoningEffort;
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
     * Returns the extra body.
     *
     * @return the extra body
     */
    public Object getExtraBody() {
        return extraBody;
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
        return builder().addMessage(Message.text(text));
    }

    /**
     * Creates a builder with the specified image url.
     *
     * @param imageUrl the image url
     * @return a new builder
     */
    public static Builder image(String imageUrl) {
        return builder().addImage(imageUrl);
    }

    /**
     * Creates a builder with the specified image data.
     *
     * @param image the image binary data
     * @param mimeType the mime type of the image
     * @return a new builder
     */
    public static Builder image(byte[] image, String mimeType) {
        return builder().addImage(image, mimeType);
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
        Type inputType = Type.CHAT_COMPLETION;
        String model;
        List<Message> messages = new ArrayList<>();
        Float frequencyPenalty;
        Map<String, Double> logitBias;
        Boolean logprobs;
        Integer topLogprobs;
        Integer maxCompletionTokens;
        Integer n;
        Float presencePenalty;
        String reasoningEffort;
        Integer seed;
        List<String> stop;
        Boolean stream;
        Float temperature;
        Float topP;
        String user;
        Boolean ignoreEos;
        List<Tool> tools;
        Object toolChoice;
        Object extraBody;

        /**
         * Sets the input type.
         *
         * @param inputType the model
         * @return the builder
         */
        public Builder inputType(Type inputType) {
            this.inputType = inputType;
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
         * Adds the message.
         *
         * @param message the message
         * @return the builder
         */
        public Builder addMessage(Message.Builder message) {
            return addMessage(message.build());
        }

        /**
         * Adds the text content.
         *
         * @param text the text
         * @return the builder
         */
        public Builder addText(String text) {
            this.messages.add(Message.text(text).build());
            return this;
        }

        /**
         * Adds the image content.
         *
         * @param imageUrl the image url
         * @return the builder
         */
        public Builder addImage(String imageUrl) {
            this.messages.add(Message.image(imageUrl).build());
            return this;
        }

        /**
         * Adds the image content.
         *
         * @param image the image data
         * @param mimeType the mime type of the image
         * @return the builder
         */
        public Builder addImage(byte[] image, String mimeType) {
            this.messages.add(Message.image(image, mimeType).build());
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
            this.messages.add(Message.file(id, data, fileName).build());
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
         * Sets the reasoning effort.
         *
         * @param reasoningEffort the reasoning effort
         * @return the builder
         */
        public Builder reasoningEffort(String reasoningEffort) {
            this.reasoningEffort = reasoningEffort;
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
         * Sets the tools.
         *
         * @param tools the tools
         * @return the tools
         */
        public Builder tools(Tool... tools) {
            return tools(Arrays.asList(tools));
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
         * Sets the extra body.
         *
         * @param extraBody the extra body
         * @return the builder
         */
        public Builder extraBody(Object extraBody) {
            this.extraBody = extraBody;
            return this;
        }

        /**
         * Builds the {@code ChatInput} instance.
         *
         * @return the {@code ChatInput} instance
         */
        public ChatInput build() {
            return new ChatInput(this);
        }
    }

    /** The target model server input schema type. */
    public enum Type {
        CHAT_COMPLETION,
        GEMINI,
        ANTHROPIC,
        ANTHROPIC_VERTEX,
    }
}
