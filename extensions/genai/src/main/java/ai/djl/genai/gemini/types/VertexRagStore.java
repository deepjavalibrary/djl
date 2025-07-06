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

/** A data class represents Gemini schema. */
@SuppressWarnings("MissingJavadocMethod")
public class VertexRagStore {

    private List<String> ragCorpora;
    private List<VertexRagStoreRagResource> ragResources;
    private RagRetrievalConfig ragRetrievalConfig;
    private Integer similarityTopK;
    private Boolean storeContext;
    private Double vectorDistanceThreshold;

    VertexRagStore(Builder builder) {
        ragCorpora = builder.ragCorpora;
        ragResources = builder.ragResources;
        ragRetrievalConfig = builder.ragRetrievalConfig;
        similarityTopK = builder.similarityTopK;
        storeContext = builder.storeContext;
        vectorDistanceThreshold = builder.vectorDistanceThreshold;
    }

    public List<String> getRagCorpora() {
        return ragCorpora;
    }

    public List<VertexRagStoreRagResource> getRagResources() {
        return ragResources;
    }

    public RagRetrievalConfig getRagRetrievalConfig() {
        return ragRetrievalConfig;
    }

    public Integer getSimilarityTopK() {
        return similarityTopK;
    }

    public Boolean getStoreContext() {
        return storeContext;
    }

    public Double getVectorDistanceThreshold() {
        return vectorDistanceThreshold;
    }

    public static Builder builder() {
        return new Builder();
    }

    /** Builder class for {@code VertexRagStore}. */
    public static final class Builder {

        List<String> ragCorpora;
        List<VertexRagStoreRagResource> ragResources = new ArrayList<>();
        RagRetrievalConfig ragRetrievalConfig;
        Integer similarityTopK;
        Boolean storeContext;
        Double vectorDistanceThreshold;

        public Builder ragCorpora(List<String> ragCorpora) {
            this.ragCorpora = ragCorpora;
            return this;
        }

        public Builder ragResources(List<VertexRagStoreRagResource> ragResources) {
            this.ragResources.clear();
            this.ragResources.addAll(ragResources);
            return this;
        }

        public Builder addRagResource(VertexRagStoreRagResource ragResource) {
            this.ragResources.add(ragResource);
            return this;
        }

        public Builder addRagResource(VertexRagStoreRagResource.Builder ragResource) {
            this.ragResources.add(ragResource.build());
            return this;
        }

        public Builder ragRetrievalConfig(RagRetrievalConfig ragRetrievalConfig) {
            this.ragRetrievalConfig = ragRetrievalConfig;
            return this;
        }

        public Builder ragRetrievalConfig(RagRetrievalConfig.Builder ragRetrievalConfig) {
            this.ragRetrievalConfig = ragRetrievalConfig.build();
            return this;
        }

        public Builder similarityTopK(Integer similarityTopK) {
            this.similarityTopK = similarityTopK;
            return this;
        }

        public Builder storeContext(Boolean storeContext) {
            this.storeContext = storeContext;
            return this;
        }

        public Builder vectorDistanceThreshold(Double vectorDistanceThreshold) {
            this.vectorDistanceThreshold = vectorDistanceThreshold;
            return this;
        }

        public VertexRagStore build() {
            return new VertexRagStore(this);
        }
    }
}
