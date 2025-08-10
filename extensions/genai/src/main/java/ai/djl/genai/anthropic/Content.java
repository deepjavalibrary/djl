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

import java.util.Base64;

/** A data class represents Anthropic schema. */
@SuppressWarnings("MissingJavadocMethod")
public class Content {

    private String type;
    private String text;
    private Source source;
    private String signature;
    private String thinking;
    private String id;
    private String name;
    private Object input;

    @SerializedName("tool_use_id")
    private String toolUseId;

    private WebSearchContent content;
    private Integer index;

    Content(Builder builder) {
        this.type = builder.type;
        this.text = builder.text;
        this.source = builder.source;
        this.signature = builder.signature;
        this.thinking = builder.thinking;
        this.id = builder.id;
        this.name = builder.name;
        this.input = builder.input;
        this.toolUseId = builder.toolUseId;
        this.content = builder.content;
        this.index = builder.index;
    }

    public String getType() {
        return type;
    }

    public String getText() {
        return text;
    }

    public Source getSource() {
        return source;
    }

    public String getSignature() {
        return signature;
    }

    public String getThinking() {
        return thinking;
    }

    public String getId() {
        return id;
    }

    public String getName() {
        return name;
    }

    public Object getInput() {
        return input;
    }

    public String getToolUseId() {
        return toolUseId;
    }

    public WebSearchContent getContent() {
        return content;
    }

    public Integer getIndex() {
        return index;
    }

    /**
     * Creates a builder to build a {@code Content}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    public static Builder text(String text) {
        return builder().type("text").text(text);
    }

    public static Builder image(String imageUrl) {
        Source.Builder sb = Source.builder().type("url").url(imageUrl);
        return builder().type("image").source(sb.build());
    }

    public static Builder image(byte[] image, String mimeType) {
        String data = Base64.getEncoder().encodeToString(image);
        Source.Builder sb = Source.builder().type("base64").mediaType(mimeType).data(data);
        return builder().type("image").source(sb.build());
    }

    /** The builder for {@code Content}. */
    public static final class Builder {

        String type;
        String text;
        Source source;
        String signature;
        String thinking;
        String id;
        String name;
        Object input;
        String toolUseId;
        WebSearchContent content;
        Integer index;

        public Builder type(String type) {
            this.type = type;
            return this;
        }

        public Builder text(String text) {
            this.text = text;
            return this;
        }

        public Builder source(Source source) {
            this.source = source;
            return this;
        }

        public Builder source(Source.Builder source) {
            return source(source.build());
        }

        public Builder signature(String signature) {
            this.signature = signature;
            return this;
        }

        public Builder thinking(String thinking) {
            this.thinking = thinking;
            return this;
        }

        public Builder id(String id) {
            this.id = id;
            return this;
        }

        public Builder name(String name) {
            this.name = name;
            return this;
        }

        public Builder input(Object input) {
            this.input = input;
            return this;
        }

        public Builder toolUseId(String toolUseId) {
            this.toolUseId = toolUseId;
            return this;
        }

        public Builder content(WebSearchContent content) {
            this.content = content;
            return this;
        }

        public Builder index(Integer index) {
            this.index = index;
            return this;
        }

        public Content build() {
            return new Content(this);
        }
    }
}
