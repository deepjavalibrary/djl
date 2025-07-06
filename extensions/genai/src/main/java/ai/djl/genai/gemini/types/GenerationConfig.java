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
package ai.djl.genai.gemini.types;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/** A data class represents Gemini schema. */
@SuppressWarnings("MissingJavadocMethod")
public class GenerationConfig {

    private Boolean audioTimestamp;
    private transient String cachedContent;
    private Integer candidateCount;
    private Float frequencyPenalty;
    private Map<String, String> labels;
    private Integer logprobs;
    private Integer maxOutputTokens;
    private MediaResolution mediaResolution;
    private ModelSelectionConfig modelSelectionConfig;
    private Float presencePenalty;
    private Object responseJsonSchema;
    private Boolean responseLogprobs;
    private String responseMimeType;
    private List<String> responseModalities;
    private Schema responseSchema;
    private transient List<SafetySetting> safetySettings;
    private Integer seed;
    private SpeechConfig speechConfig;
    private List<String> stopSequences;
    private transient Content systemInstruction;
    private Float temperature;
    private ThinkingConfig thinkingConfig;
    private transient ToolConfig toolConfig;
    private transient List<Tool> tools;
    private Float topK;
    private Float topP;

    GenerationConfig(Builder builder) {
        audioTimestamp = builder.audioTimestamp;
        cachedContent = builder.cachedContent;
        candidateCount = builder.candidateCount;
        frequencyPenalty = builder.frequencyPenalty;
        labels = builder.labels;
        logprobs = builder.logprobs;
        maxOutputTokens = builder.maxOutputTokens;
        mediaResolution = builder.mediaResolution;
        modelSelectionConfig = builder.modelSelectionConfig;
        presencePenalty = builder.presencePenalty;
        responseJsonSchema = builder.responseJsonSchema;
        responseLogprobs = builder.responseLogprobs;
        responseMimeType = builder.responseMimeType;
        responseModalities = builder.responseModalities;
        responseSchema = builder.responseSchema;
        safetySettings = builder.safetySettings;
        seed = builder.seed;
        speechConfig = builder.speechConfig;
        stopSequences = builder.stopSequences;
        systemInstruction = builder.systemInstruction;
        temperature = builder.temperature;
        thinkingConfig = builder.thinkingConfig;
        toolConfig = builder.toolConfig;
        tools = builder.tools;
        topK = builder.topK;
        topP = builder.topP;
    }

    public Boolean getAudioTimestamp() {
        return audioTimestamp;
    }

    public String getCachedContent() {
        return cachedContent;
    }

    public Integer getCandidateCount() {
        return candidateCount;
    }

    public Float getFrequencyPenalty() {
        return frequencyPenalty;
    }

    public Map<String, String> getLabels() {
        return labels;
    }

    public Integer getLogprobs() {
        return logprobs;
    }

    public Integer getMaxOutputTokens() {
        return maxOutputTokens;
    }

    public MediaResolution getMediaResolution() {
        return mediaResolution;
    }

    public ModelSelectionConfig getModelSelectionConfig() {
        return modelSelectionConfig;
    }

    public Float getPresencePenalty() {
        return presencePenalty;
    }

    public Object getResponseJsonSchema() {
        return responseJsonSchema;
    }

    public Boolean getResponseLogprobs() {
        return responseLogprobs;
    }

    public String getResponseMimeType() {
        return responseMimeType;
    }

    public List<String> getResponseModalities() {
        return responseModalities;
    }

    public Schema getResponseSchema() {
        return responseSchema;
    }

    public List<SafetySetting> getSafetySettings() {
        return safetySettings;
    }

    public Integer getSeed() {
        return seed;
    }

    public SpeechConfig getSpeechConfig() {
        return speechConfig;
    }

    public List<String> getStopSequences() {
        return stopSequences;
    }

    public Content getSystemInstruction() {
        return systemInstruction;
    }

    public Float getTemperature() {
        return temperature;
    }

    public ThinkingConfig getThinkingConfig() {
        return thinkingConfig;
    }

    public ToolConfig getToolConfig() {
        return toolConfig;
    }

    public List<Tool> getTools() {
        return tools;
    }

    public Float getTopK() {
        return topK;
    }

    public Float getTopP() {
        return topP;
    }

    public static Builder builder() {
        return new Builder();
    }

    /** Builder class for {@code GenerationConfig}. */
    public static final class Builder {

        Boolean audioTimestamp;
        String cachedContent;
        Integer candidateCount;
        Float frequencyPenalty;
        Map<String, String> labels;
        Integer logprobs;
        Integer maxOutputTokens;
        MediaResolution mediaResolution;
        ModelSelectionConfig modelSelectionConfig;
        Float presencePenalty;
        Object responseJsonSchema;
        Boolean responseLogprobs;
        String responseMimeType;
        List<String> responseModalities;
        Schema responseSchema;
        List<SafetySetting> safetySettings = new ArrayList<>();
        Integer seed;
        SpeechConfig speechConfig;
        List<String> stopSequences;
        Content systemInstruction;
        Float temperature;
        ThinkingConfig thinkingConfig;
        ToolConfig toolConfig;
        List<Tool> tools = new ArrayList<>();
        Float topK;
        Float topP;

        public Builder audioTimestamp(Boolean audioTimestamp) {
            this.audioTimestamp = audioTimestamp;
            return this;
        }

        public Builder cachedContent(String cachedContent) {
            this.cachedContent = cachedContent;
            return this;
        }

        public Builder candidateCount(Integer candidateCount) {
            this.candidateCount = candidateCount;
            return this;
        }

        public Builder frequencyPenalty(Float frequencyPenalty) {
            this.frequencyPenalty = frequencyPenalty;
            return this;
        }

        public Builder labels(Map<String, String> labels) {
            this.labels = labels;
            return this;
        }

        public Builder logprobs(Integer logprobs) {
            this.logprobs = logprobs;
            return this;
        }

        public Builder maxOutputTokens(Integer maxOutputTokens) {
            this.maxOutputTokens = maxOutputTokens;
            return this;
        }

        public Builder mediaResolution(MediaResolution mediaResolution) {
            this.mediaResolution = mediaResolution;
            return this;
        }

        public Builder modelSelectionConfig(ModelSelectionConfig modelSelectionConfig) {
            this.modelSelectionConfig = modelSelectionConfig;
            return this;
        }

        public Builder modelSelectionConfig(ModelSelectionConfig.Builder modelSelectionConfig) {
            this.modelSelectionConfig = modelSelectionConfig.build();
            return this;
        }

        public Builder presencePenalty(Float presencePenalty) {
            this.presencePenalty = presencePenalty;
            return this;
        }

        public Builder responseJsonSchema(Object responseJsonSchema) {
            this.responseJsonSchema = responseJsonSchema;
            return this;
        }

        public Builder responseLogprobs(Boolean responseLogprobs) {
            this.responseLogprobs = responseLogprobs;
            return this;
        }

        public Builder responseMimeType(String responseMimeType) {
            this.responseMimeType = responseMimeType;
            return this;
        }

        public Builder responseModalities(List<String> responseModalities) {
            this.responseModalities = responseModalities;
            return this;
        }

        public Builder responseSchema(Schema responseSchema) {
            this.responseSchema = responseSchema;
            return this;
        }

        public Builder responseSchema(Schema.Builder responseSchema) {
            this.responseSchema = responseSchema.build();
            return this;
        }

        public Builder safetySettings(List<SafetySetting> safetySettings) {
            this.safetySettings.clear();
            this.safetySettings.addAll(safetySettings);
            return this;
        }

        public Builder addSafetySetting(SafetySetting safetySetting) {
            this.safetySettings.add(safetySetting);
            return this;
        }

        public Builder addSafetySetting(SafetySetting.Builder safetySetting) {
            this.safetySettings.add(safetySetting.build());
            return this;
        }

        public Builder seed(Integer seed) {
            this.seed = seed;
            return this;
        }

        public Builder speechConfig(SpeechConfig speechConfig) {
            this.speechConfig = speechConfig;
            return this;
        }

        public Builder speechConfig(SpeechConfig.Builder speechConfig) {
            this.speechConfig = speechConfig.build();
            return this;
        }

        public Builder stopSequences(List<String> stopSequences) {
            this.stopSequences = stopSequences;
            return this;
        }

        public Builder systemInstruction(Content systemInstruction) {
            this.systemInstruction = systemInstruction;
            return this;
        }

        public Builder systemInstruction(Content.Builder systemInstruction) {
            this.systemInstruction = systemInstruction.build();
            return this;
        }

        public Builder systemInstruction(String systemInstruction) {
            return this.systemInstruction(Content.text(systemInstruction));
        }

        public Builder temperature(Float temperature) {
            this.temperature = temperature;
            return this;
        }

        public Builder thinkingConfig(ThinkingConfig thinkingConfig) {
            this.thinkingConfig = thinkingConfig;
            return this;
        }

        public Builder thinkingConfig(ThinkingConfig.Builder thinkingConfig) {
            this.thinkingConfig = thinkingConfig.build();
            return this;
        }

        public Builder toolConfig(ToolConfig toolConfig) {
            this.toolConfig = toolConfig;
            return this;
        }

        public Builder toolConfig(ToolConfig.Builder toolConfig) {
            this.toolConfig = toolConfig.build();
            return this;
        }

        public Builder tools(List<Tool> tools) {
            this.tools.clear();
            this.tools.addAll(tools);
            return this;
        }

        public Builder addTool(Tool tool) {
            this.tools.add(tool);
            return this;
        }

        public Builder addTool(Tool.Builder tool) {
            this.tools.add(tool.build());
            return this;
        }

        public Builder topK(Float topK) {
            this.topK = topK;
            return this;
        }

        public Builder topP(Float topP) {
            this.topP = topP;
            return this;
        }

        public GenerationConfig build() {
            return new GenerationConfig(this);
        }
    }
}
