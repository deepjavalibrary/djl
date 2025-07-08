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
public class GroundingChunkRetrievedContext {

    private RagChunk ragChunk;
    private String text;
    private String title;
    private String uri;

    GroundingChunkRetrievedContext(Builder builder) {
        ragChunk = builder.ragChunk;
        text = builder.text;
        title = builder.title;
        uri = builder.uri;
    }

    public RagChunk getRagChunk() {
        return ragChunk;
    }

    public String getText() {
        return text;
    }

    public String getTitle() {
        return title;
    }

    public String getUri() {
        return uri;
    }

    public static Builder builder() {
        return new Builder();
    }

    /** Builder class for {@code GroundingChunkRetrievedContext}. */
    public static final class Builder {

        RagChunk ragChunk;
        String text;
        String title;
        String uri;

        public Builder ragChunk(RagChunk ragChunk) {
            this.ragChunk = ragChunk;
            return this;
        }

        public Builder ragChunk(RagChunk.Builder ragChunk) {
            this.ragChunk = ragChunk.build();
            return this;
        }

        public Builder text(String text) {
            this.text = text;
            return this;
        }

        public Builder title(String title) {
            this.title = title;
            return this;
        }

        public Builder uri(String uri) {
            this.uri = uri;
            return this;
        }

        public GroundingChunkRetrievedContext build() {
            return new GroundingChunkRetrievedContext(this);
        }
    }
}
