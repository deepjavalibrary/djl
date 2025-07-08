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

import java.util.List;

/** A data class represents Gemini schema. */
@SuppressWarnings("MissingJavadocMethod")
public class GroundingSupport {

    private List<Float> confidenceScores;
    private List<Integer> groundingChunkIndices;
    private Segment segment;

    GroundingSupport(Builder builder) {
        confidenceScores = builder.confidenceScores;
        groundingChunkIndices = builder.groundingChunkIndices;
        segment = builder.segment;
    }

    public List<Float> getConfidenceScores() {
        return confidenceScores;
    }

    public List<Integer> getGroundingChunkIndices() {
        return groundingChunkIndices;
    }

    public Segment getSegment() {
        return segment;
    }

    public static Builder builder() {
        return new Builder();
    }

    /** Builder class for {@code GroundingSupport}. */
    public static final class Builder {

        List<Float> confidenceScores;
        List<Integer> groundingChunkIndices;
        Segment segment;

        public Builder confidenceScores(List<Float> confidenceScores) {
            this.confidenceScores = confidenceScores;
            return this;
        }

        public Builder groundingChunkIndices(List<Integer> groundingChunkIndices) {
            this.groundingChunkIndices = groundingChunkIndices;
            return this;
        }

        public Builder segment(Segment segment) {
            this.segment = segment;
            return this;
        }

        public Builder segment(Segment.Builder segment) {
            this.segment = segment.build();
            return this;
        }

        public GroundingSupport build() {
            return new GroundingSupport(this);
        }
    }
}
