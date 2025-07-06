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
public class GroundingMetadata {

    private List<GroundingChunk> groundingChunks;
    private List<GroundingSupport> groundingSupports;
    private RetrievalMetadata retrievalMetadata;
    private List<String> retrievalQueries;
    private SearchEntryPoint searchEntryPoint;
    private List<String> webSearchQueries;

    GroundingMetadata(Builder builder) {
        groundingChunks = builder.groundingChunks;
        groundingSupports = builder.groundingSupports;
        retrievalMetadata = builder.retrievalMetadata;
        retrievalQueries = builder.retrievalQueries;
        searchEntryPoint = builder.searchEntryPoint;
        webSearchQueries = builder.webSearchQueries;
    }

    public List<GroundingChunk> getGroundingChunks() {
        return groundingChunks;
    }

    public List<GroundingSupport> getGroundingSupports() {
        return groundingSupports;
    }

    public RetrievalMetadata getRetrievalMetadata() {
        return retrievalMetadata;
    }

    public List<String> getRetrievalQueries() {
        return retrievalQueries;
    }

    public SearchEntryPoint getSearchEntryPoint() {
        return searchEntryPoint;
    }

    public List<String> getWebSearchQueries() {
        return webSearchQueries;
    }

    public static Builder builder() {
        return new Builder();
    }

    /** Builder class for {@code GroundingMetadata}. */
    public static final class Builder {

        List<GroundingChunk> groundingChunks = new ArrayList<>();
        List<GroundingSupport> groundingSupports = new ArrayList<>();
        RetrievalMetadata retrievalMetadata;
        List<String> retrievalQueries;
        SearchEntryPoint searchEntryPoint;
        List<String> webSearchQueries;

        public Builder groundingChunks(List<GroundingChunk> groundingChunks) {
            this.groundingChunks.clear();
            this.groundingChunks.addAll(groundingChunks);
            return this;
        }

        public Builder addGroundingChunk(GroundingChunk groundingChunk) {
            this.groundingChunks.add(groundingChunk);
            return this;
        }

        public Builder addGroundingChunk(GroundingChunk.Builder groundingChunk) {
            this.groundingChunks.add(groundingChunk.build());
            return this;
        }

        public Builder groundingSupports(List<GroundingSupport> groundingSupports) {
            this.groundingSupports.clear();
            this.groundingSupports.addAll(groundingSupports);
            return this;
        }

        public Builder addGroundingSupport(GroundingSupport groundingSupport) {
            this.groundingSupports.add(groundingSupport);
            return this;
        }

        public Builder addGroundingSupport(GroundingSupport.Builder groundingSupport) {
            this.groundingSupports.add(groundingSupport.build());
            return this;
        }

        public Builder retrievalMetadata(RetrievalMetadata retrievalMetadata) {
            this.retrievalMetadata = retrievalMetadata;
            return this;
        }

        public Builder retrievalMetadata(RetrievalMetadata.Builder retrievalMetadata) {
            this.retrievalMetadata = retrievalMetadata.build();
            return this;
        }

        public Builder retrievalQueries(List<String> retrievalQueries) {
            this.retrievalQueries = retrievalQueries;
            return this;
        }

        public Builder searchEntryPoint(SearchEntryPoint searchEntryPoint) {
            this.searchEntryPoint = searchEntryPoint;
            return this;
        }

        public Builder searchEntryPoint(SearchEntryPoint.Builder searchEntryPoint) {
            this.searchEntryPoint = searchEntryPoint.build();
            return this;
        }

        public Builder webSearchQueries(List<String> webSearchQueries) {
            this.webSearchQueries = webSearchQueries;
            return this;
        }

        public GroundingMetadata build() {
            return new GroundingMetadata(this);
        }
    }
}
