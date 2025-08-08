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

import java.util.ArrayList;
import java.util.List;

/** A data class represents Anthropic schema. */
@SuppressWarnings("MissingJavadocMethod")
public class Message {

    private String role;
    private List<Content> content;

    Message(Builder builder) {
        this.role = builder.role;
        this.content = builder.content;
    }

    public String getRole() {
        return role;
    }

    public List<Content> getContent() {
        return content;
    }

    /**
     * Creates a builder to build a {@code Message}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    public static Builder text(String text) {
        return builder().addContent(Content.text(text));
    }

    public static Builder image(byte[] image, String mimeType) {
        return builder().addContent(Content.image(image, mimeType));
    }

    public static Builder image(String imageUrl) {
        return builder().addContent(Content.image(imageUrl));
    }

    /** The builder for {@code Message}. */
    public static final class Builder {

        String role = "user";
        List<Content> content;

        public Builder role(String role) {
            this.role = role;
            return this;
        }

        public Builder content(List<Content> content) {
            this.content = content;
            return this;
        }

        public Builder addContent(Content.Builder content) {
            return addContent(content.build());
        }

        public Builder addContent(Content content) {
            if (this.content == null) {
                this.content = new ArrayList<>();
            }
            this.content.add(content);
            return this;
        }

        public Builder text(String text) {
            return addContent(Content.text(text));
        }

        public Builder image(String imageUrl) {
            return addContent(Content.image(imageUrl));
        }

        public Builder image(byte[] image, String mimeType) {
            return addContent(Content.image(image, mimeType));
        }

        public Message build() {
            return new Message(this);
        }
    }
}
