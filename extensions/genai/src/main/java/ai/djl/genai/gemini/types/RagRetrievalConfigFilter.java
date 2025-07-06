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
public class RagRetrievalConfigFilter {

    private String metadataFilter;
    private Double vectorDistanceThreshold;
    private Double vectorSimilarityThreshold;

    RagRetrievalConfigFilter(Builder builder) {
        metadataFilter = builder.metadataFilter;
        vectorDistanceThreshold = builder.vectorDistanceThreshold;
        vectorSimilarityThreshold = builder.vectorSimilarityThreshold;
    }

    public String getMetadataFilter() {
        return metadataFilter;
    }

    public Double getVectorDistanceThreshold() {
        return vectorDistanceThreshold;
    }

    public Double getVectorSimilarityThreshold() {
        return vectorSimilarityThreshold;
    }

    public static Builder builder() {
        return new Builder();
    }

    /** Builder class for {@code RagRetrievalConfigFilter}. */
    public static final class Builder {

        String metadataFilter;
        Double vectorDistanceThreshold;
        Double vectorSimilarityThreshold;

        public Builder metadataFilter(String metadataFilter) {
            this.metadataFilter = metadataFilter;
            return this;
        }

        public Builder vectorDistanceThreshold(Double vectorDistanceThreshold) {
            this.vectorDistanceThreshold = vectorDistanceThreshold;
            return this;
        }

        public Builder vectorSimilarityThreshold(Double vectorSimilarityThreshold) {
            this.vectorSimilarityThreshold = vectorSimilarityThreshold;
            return this;
        }

        public RagRetrievalConfigFilter build() {
            return new RagRetrievalConfigFilter(this);
        }
    }
}
