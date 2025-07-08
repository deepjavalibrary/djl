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
package ai.djl.genai.gemini;

import ai.djl.genai.gemini.types.Blob;
import ai.djl.genai.gemini.types.Content;
import ai.djl.genai.gemini.types.FileData;
import ai.djl.genai.gemini.types.GenerationConfig;
import ai.djl.genai.gemini.types.Part;
import ai.djl.genai.gemini.types.SafetySetting;
import ai.djl.genai.gemini.types.Tool;
import ai.djl.genai.gemini.types.ToolConfig;

import java.util.ArrayList;
import java.util.List;

/** A class presents the Gemini input. */
public class GeminiInput {

    private List<Content> contents;
    private GenerationConfig generationConfig;
    private Content systemInstruction;
    private List<SafetySetting> safetySettings;
    private List<Tool> tools;
    private ToolConfig toolConfig;
    private String cachedContent;

    GeminiInput(Builder builder) {
        this.contents = builder.contents;
        this.generationConfig = builder.generationConfig;
        this.systemInstruction = builder.systemInstruction;
        this.safetySettings = builder.safetySettings;
        this.tools = builder.tools;
        this.toolConfig = builder.toolConfig;
        this.cachedContent = builder.cachedContent;
    }

    /**
     * Returns the contents.
     *
     * @return the contents
     */
    public List<Content> getContents() {
        return contents;
    }

    /**
     * Returns the generation config.
     *
     * @return the generation config
     */
    public GenerationConfig getGenerationConfig() {
        return generationConfig;
    }

    /**
     * Returns the system instruction.
     *
     * @return the system instruction
     */
    public Content getSystemInstruction() {
        return systemInstruction;
    }

    /**
     * Returns the safety settings.
     *
     * @return the safety settings
     */
    public List<SafetySetting> getSafetySettings() {
        return safetySettings;
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
     * Returns the toolConfig.
     *
     * @return the toolConfig
     */
    public ToolConfig getToolConfig() {
        return toolConfig;
    }

    /**
     * Returns the cachedContent.
     *
     * @return the cachedContent
     */
    public String getCachedContent() {
        return cachedContent;
    }

    /**
     * Creates a builder to build a {@code GeminiInput}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Creates a builder with text content.
     *
     * @param text the text content
     * @return a new builder
     */
    public static Builder text(String text) {
        return text(text, null);
    }

    /**
     * Creates a builder with text content and configuration.
     *
     * @param text the text content
     * @param config the generation config
     * @return a new builder
     */
    public static Builder text(String text, GenerationConfig config) {
        return builder().addText(text).generationConfig(config);
    }

    /**
     * Creates a builder with the file uri content.
     *
     * @param text the file uri
     * @param mimeType the mime type
     * @return a new builder
     */
    public static Builder fileUri(String text, String mimeType) {
        return builder().addUri(text, mimeType);
    }

    /**
     * Creates a builder with the inline data content.
     *
     * @param bytes the bytes of the inline data
     * @param mimeType the mime type
     * @return a new builder
     */
    public static Builder bytes(byte[] bytes, String mimeType) {
        return builder().addBytes(bytes, mimeType);
    }

    /** The builder for {@code GeminiInput}. */
    public static final class Builder {

        List<Content> contents = new ArrayList<>();
        GenerationConfig generationConfig;
        Content systemInstruction;
        List<SafetySetting> safetySettings;
        List<Tool> tools;
        ToolConfig toolConfig;

        String cachedContent;

        /**
         * Sets the contents.
         *
         * @param contents the contents
         * @return the builder
         */
        public Builder contents(List<Content> contents) {
            this.contents.clear();
            this.contents.addAll(contents);
            return this;
        }

        /**
         * Adds the content.
         *
         * @param content the content
         * @return the builder
         */
        public Builder addContent(Content content) {
            contents.add(content);
            return this;
        }

        /**
         * Adds the content.
         *
         * @param content the content builder
         * @return the builder
         */
        public Builder addContent(Content.Builder content) {
            return addContent(content.build());
        }

        /**
         * Adds the text content.
         *
         * @param text the text
         * @return the builder
         */
        public Builder addText(String text) {
            return addText(text, "user");
        }

        /**
         * Sets the text content with the role.
         *
         * @param text the text
         * @param role the role
         * @return the builder
         */
        public Builder addText(String text, String role) {
            return addContent(Content.text(text).role(role));
        }

        /**
         * Sets the model.
         *
         * @param fileUri the fileUri
         * @param mimeType the mimeType
         * @return the builder
         */
        public Builder addUri(String fileUri, String mimeType) {
            FileData.Builder file = FileData.builder().fileUri(fileUri).mimeType(mimeType);
            Part.Builder part = Part.builder().fileData(file);
            return addContent(Content.builder().addPart(part).role("user"));
        }

        /**
         * Sets the inline data.
         *
         * @param bytes the bytes
         * @param mimeType the mimeType
         * @return the builder
         */
        public Builder addBytes(byte[] bytes, String mimeType) {
            Blob.Builder file = Blob.builder().data(bytes).mimeType(mimeType);
            Part.Builder part = Part.builder().inlineData(file);
            return addContent(Content.builder().addPart(part).role("user"));
        }

        /**
         * Sets the generation config.
         *
         * @param generationConfig the generationConfig
         * @return the builder
         */
        public Builder generationConfig(GenerationConfig generationConfig) {
            this.generationConfig = generationConfig;
            if (generationConfig != null) {
                this.cachedContent = generationConfig.getCachedContent();
                this.safetySettings = generationConfig.getSafetySettings();
                this.systemInstruction = generationConfig.getSystemInstruction();
                this.toolConfig = generationConfig.getToolConfig();
                this.tools = generationConfig.getTools();
            }
            return this;
        }

        /**
         * Returns the {@code GeminiInput} instance.
         *
         * @return the {@code GeminiInput} instance
         */
        public GeminiInput build() {
            return new GeminiInput(this);
        }
    }
}
