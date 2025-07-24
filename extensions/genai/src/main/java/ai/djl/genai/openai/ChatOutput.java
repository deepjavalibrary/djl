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

import ai.djl.genai.gemini.GeminiOutput;
import ai.djl.genai.gemini.types.Blob;
import ai.djl.genai.gemini.types.Candidate;
import ai.djl.genai.gemini.types.FileData;
import ai.djl.genai.gemini.types.FinishReason;
import ai.djl.genai.gemini.types.FunctionCall;
import ai.djl.genai.gemini.types.LogprobsResult;
import ai.djl.genai.gemini.types.LogprobsResultCandidate;
import ai.djl.genai.gemini.types.LogprobsResultTopCandidates;
import ai.djl.genai.gemini.types.Part;
import ai.djl.genai.gemini.types.UsageMetadata;
import ai.djl.util.JsonUtils;

import com.google.gson.JsonObject;

import java.time.Instant;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/** The chat completion style output. */
public class ChatOutput {

    private String id;
    private String object;
    private Long created;
    private List<Choice> choices;
    private String model;
    private Usage usage;

    ChatOutput() {}

    ChatOutput(
            String id,
            String object,
            Long created,
            List<Choice> choices,
            String model,
            Usage usage) {
        this.id = id;
        this.object = object;
        this.created = created;
        this.choices = choices;
        this.model = model;
        this.usage = usage;
    }

    /**
     * Returns the id.
     *
     * @return the id
     */
    public String getId() {
        return id;
    }

    /**
     * Returns the object.
     *
     * @return the object
     */
    public String getObject() {
        return object;
    }

    /**
     * Returns the created time.
     *
     * @return the created time
     */
    public Long getCreated() {
        return created;
    }

    /**
     * Returns the choices.
     *
     * @return the choices
     */
    public List<Choice> getChoices() {
        return choices;
    }

    /**
     * Returns the model name.
     *
     * @return the model name
     */
    public String getModel() {
        return model;
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
     * Returns the aggregated text output.
     *
     * @return the aggregated text output
     */
    @SuppressWarnings("unchecked")
    public String getTextOutput() {
        if (choices == null || choices.isEmpty()) {
            return "";
        }
        Message message = choices.get(0).getMessage();
        if (message == null) {
            message = choices.get(0).getDelta();
        }
        if (message == null) {
            return "";
        }
        Object content = message.getContent();
        if (content instanceof String) {
            return (String) content;
        } else if (content instanceof List) {
            List<Content> contents = (List<Content>) content;
            StringBuilder sb = new StringBuilder();
            for (Content part : contents) {
                if ("text".equals(part.getType())) {
                    sb.append(part.getText());
                }
            }
            return sb.toString();
        }
        return "";
    }

    /**
     * Returns the {@link ToolCall} response.
     *
     * @return the {@link ToolCall} response
     */
    public ToolCall getToolCall() {
        if (choices != null && !choices.isEmpty()) {
            Message message = choices.get(0).getMessage();
            if (message != null) {
                List<ToolCall> toolsCalls = message.getToolCalls();
                if (toolsCalls != null && !toolsCalls.isEmpty()) {
                    return toolsCalls.get(0);
                }
            }
        }
        return null;
    }

    /**
     * Returns the per token log probability.
     *
     * @return the per token log probability
     */
    public List<Logprob> getLogprobs() {
        if (choices != null && !choices.isEmpty()) {
            List<Logprob> logprobs = choices.get(0).getLogprobs();
            if (logprobs != null) {
                return logprobs;
            }
        }
        return Collections.emptyList();
    }

    /**
     * Customizes schema deserialization.
     *
     * @param json the output json string
     * @return the deserialized {@code ChatOutput} instance
     */
    public static ChatOutput fromJson(String json) {
        JsonObject element = JsonUtils.GSON.fromJson(json, JsonObject.class);
        if (element.has("candidates")) {
            GeminiOutput gemini = JsonUtils.GSON.fromJson(element, GeminiOutput.class);
            return fromGemini(gemini);
        }
        return ChatInput.GSON.fromJson(element, ChatOutput.class);
    }

    static ChatOutput fromGemini(GeminiOutput gemini) {
        String id = gemini.getResponseId();
        String create = gemini.getCreateTime();
        Long time = null;
        if (create != null) {
            time = Instant.parse(create).toEpochMilli();
        }

        Usage usage = null;
        UsageMetadata um = gemini.getUsageMetadata();
        if (um != null) {
            usage =
                    new Usage(
                            um.getCandidatesTokenCount(),
                            um.getPromptTokenCount(),
                            um.getTotalTokenCount());
        }
        String model = gemini.getModelVersion();

        List<Candidate> candidates = gemini.getCandidates();
        List<Choice> choices = new ArrayList<>(candidates.size());
        for (Candidate candidate : candidates) {
            ai.djl.genai.gemini.types.Content content = candidate.getContent();
            String role = content.getRole();
            List<Part> parts = content.getParts();
            Message.Builder message = Message.builder().role(role);
            if (parts != null) {
                for (Part part : parts) {
                    String text = part.getText();
                    Blob inline = part.getInlineData();
                    FileData fileData = part.getFileData();
                    FunctionCall func = part.getFunctionCall();
                    if (text != null) {
                        message.addText(text);
                    } else if (inline != null) {
                        String url = "data:" + inline.getMimeType() + ";base64," + inline.getData();
                        message.addContent(new Content(new Content.ImageContent(url)));
                    } else if (fileData != null) {
                        String fileUri = fileData.getFileUri();
                        String fileName = fileData.getDisplayName();
                        message.addContent(
                                new Content(new Content.FileContent(fileUri, null, fileName)));
                    } else if (func != null) {
                        String callId = func.getId();
                        String args = JsonUtils.GSON_COMPACT.toJson(func.getArgs());
                        ToolCall.Function function = new ToolCall.Function(args, func.getName());
                        ToolCall toolCall = new ToolCall(callId, "function", function);
                        message.toolCalls(toolCall).toolCallId(callId);
                    }
                }
            }
            List<Logprob> logprobs = null;
            LogprobsResult lr = candidate.getLogprobsResult();
            if (lr != null) {
                List<LogprobsResultCandidate> lrcs = lr.getChosenCandidates();
                List<LogprobsResultTopCandidates> tlcs = lr.getTopCandidates();
                logprobs = new ArrayList<>();
                int index = 0;
                for (LogprobsResultCandidate lrc : lrcs) {
                    List<TopLogprob> topLogprobs = null;
                    if (tlcs != null && index < tlcs.size()) {
                        topLogprobs = new ArrayList<>();
                        for (LogprobsResultCandidate tlc : tlcs.get(index).getCandidates()) {
                            topLogprobs.add(
                                    new TopLogprob(tlc.getToken(), tlc.getLogProbability(), null));
                        }
                    }
                    logprobs.add(
                            new Logprob(
                                    lrc.getToken(), lrc.getLogProbability(), null, topLogprobs));
                    ++index;
                }
            }

            FinishReason reason = candidate.getFinishReason();
            Choice choice =
                    new Choice(
                            candidate.getIndex(), message.build(), logprobs, null, reason.name());
            choices.add(choice);
        }

        return new ChatOutput(id, "chat.completion", time, choices, model, usage);
    }
}
