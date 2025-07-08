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
public class VideoMetadata {

    private String endOffset;
    private Double fps;
    private String startOffset;

    VideoMetadata(Builder builder) {
        endOffset = builder.endOffset;
        fps = builder.fps;
        startOffset = builder.startOffset;
    }

    public String getEndOffset() {
        return endOffset;
    }

    public Double getFps() {
        return fps;
    }

    public String getStartOffset() {
        return startOffset;
    }

    public static Builder builder() {
        return new Builder();
    }

    /** Builder class for {@code VideoMetadata}. */
    public static final class Builder {

        String endOffset;
        Double fps;
        String startOffset;

        public Builder endOffset(String endOffset) {
            this.endOffset = endOffset;
            return this;
        }

        public Builder fps(Double fps) {
            this.fps = fps;
            return this;
        }

        public Builder startOffset(String startOffset) {
            this.startOffset = startOffset;
            return this;
        }

        public VideoMetadata build() {
            return new VideoMetadata(this);
        }
    }
}
