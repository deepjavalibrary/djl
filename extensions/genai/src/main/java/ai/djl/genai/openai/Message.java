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

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/** A data class represents chat completion schema. */
@SuppressWarnings({"MissingJavadocMethod", "serial"})
public class Message implements Serializable {

    private static final long serialVersionUID = 1L;

    private String role;
    private Object content;
    private String name;

    @SerializedName("tool_calls")
    private List<ToolCall> toolCalls;

    @SerializedName("tool_call_id")
    private String toolCallId;

    Message(Builder builder) {
        role = builder.role;
        if (!builder.contents.isEmpty()) {
            content = builder.contents;
        } else {
            content = builder.text;
        }
        name = builder.name;
        toolCalls = builder.toolCalls;
        toolCallId = builder.toolCallId;
    }

    public String getRole() {
        return role;
    }

    public Object getContent() {
        return content;
    }

    public String getName() {
        return name;
    }

    public List<ToolCall> getToolCalls() {
        return toolCalls;
    }

    public String getToolCallId() {
        return toolCallId;
    }

    public static Builder text(String text) {
        return text(text, "user");
    }

    public static Builder text(String text, String role) {
        return builder().text(text).role(role);
    }

    public static Builder image(String imageUrl) {
        return image(imageUrl, "user");
    }

    public static Builder image(byte[] image, String mimeType) {
        return image(image, mimeType, "user");
    }

    public static Builder image(byte[] image, String mimeType, String role) {
        return builder().addImage(image, mimeType).role(role);
    }

    public static Builder image(String imageUrl, String role) {
        return builder().addImage(imageUrl).role(role);
    }

    public static Builder file(String id, byte[] data, String fileName) {
        return file(id, data, fileName, "user");
    }

    public static Builder file(String id, byte[] data, String fileName, String role) {
        return builder().addFile(id, data, fileName).role(role);
    }

    /**
     * Creates a builder to build a {@code Message}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /** The builder for {@code Message}. */
    public static final class Builder {

        String role;
        String text;
        List<Content> contents = new ArrayList<>();
        String name;
        List<ToolCall> toolCalls;
        String toolCallId;

        public Builder role(String role) {
            this.role = role;
            return this;
        }

        public Builder text(String text) {
            this.text = text;
            return this;
        }

        public Builder name(String name) {
            this.name = name;
            return this;
        }

        public Builder toolCalls(List<ToolCall> toolCalls) {
            this.toolCalls = toolCalls;
            return this;
        }

        public Builder toolCalls(ToolCall... toolCalls) {
            return toolCalls(Arrays.asList(toolCalls));
        }

        public Builder toolCallId(String toolCallId) {
            this.toolCallId = toolCallId;
            return this;
        }

        public Builder addText(String text) {
            contents.add(Content.text(text));
            return this;
        }

        public Builder addImage(String imageUrl) {
            contents.add(Content.image(imageUrl));
            return this;
        }

        public Builder addImage(byte[] image, String mimeType) {
            contents.add(Content.image(image, mimeType));
            return this;
        }

        public Builder addFile(String id, byte[] data, String fileName) {
            contents.add(Content.file(id, data, fileName));
            return this;
        }

        public Builder addContent(Content content) {
            contents.add(content);
            return this;
        }

        public Message build() {
            return new Message(this);
        }
    }
}
