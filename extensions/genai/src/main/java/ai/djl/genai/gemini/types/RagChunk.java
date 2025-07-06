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

/** A data class represents Gemini schema. */
@SuppressWarnings("MissingJavadocMethod")
public class RagChunk {

    private RagChunkPageSpan pageSpan;
    private String text;

    RagChunk(Builder builder) {
        pageSpan = builder.pageSpan;
        text = builder.text;
    }

    public RagChunkPageSpan getPageSpan() {
        return pageSpan;
    }

    public String getText() {
        return text;
    }

    public static Builder builder() {
        return new Builder();
    }

    /** Builder class for {@code RagChunk}. */
    public static final class Builder {

        RagChunkPageSpan pageSpan;
        String text;

        public Builder pageSpan(RagChunkPageSpan pageSpan) {
            this.pageSpan = pageSpan;
            return this;
        }

        public Builder pageSpan(RagChunkPageSpan.Builder pageSpan) {
            this.pageSpan = pageSpan.build();
            return this;
        }

        public Builder text(String text) {
            this.text = text;
            return this;
        }

        public RagChunk build() {
            return new RagChunk(this);
        }
    }
}
