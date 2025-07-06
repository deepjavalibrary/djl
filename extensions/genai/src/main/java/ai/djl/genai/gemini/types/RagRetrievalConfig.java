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
public class RagRetrievalConfig {

    private RagRetrievalConfigFilter filter;
    private RagRetrievalConfigHybridSearch hybridSearch;
    private RagRetrievalConfigRanking ranking;
    private Integer topK;

    RagRetrievalConfig(Builder builder) {
        filter = builder.filter;
        hybridSearch = builder.hybridSearch;
        ranking = builder.ranking;
        topK = builder.topK;
    }

    public RagRetrievalConfigFilter getFilter() {
        return filter;
    }

    public RagRetrievalConfigHybridSearch getHybridSearch() {
        return hybridSearch;
    }

    public RagRetrievalConfigRanking getRanking() {
        return ranking;
    }

    public Integer getTopK() {
        return topK;
    }

    public static Builder builder() {
        return new Builder();
    }

    /** Builder class for {@code RagRetrievalConfig}. */
    public static final class Builder {

        RagRetrievalConfigFilter filter;
        RagRetrievalConfigHybridSearch hybridSearch;
        RagRetrievalConfigRanking ranking;
        Integer topK;

        public Builder filter(RagRetrievalConfigFilter filter) {
            this.filter = filter;
            return this;
        }

        public Builder filter(RagRetrievalConfigFilter.Builder filter) {
            this.filter = filter.build();
            return this;
        }

        public Builder hybridSearch(RagRetrievalConfigHybridSearch hybridSearch) {
            this.hybridSearch = hybridSearch;
            return this;
        }

        public Builder hybridSearch(RagRetrievalConfigHybridSearch.Builder hybridSearch) {
            this.hybridSearch = hybridSearch.build();
            return this;
        }

        public Builder ranking(RagRetrievalConfigRanking ranking) {
            this.ranking = ranking;
            return this;
        }

        public Builder ranking(RagRetrievalConfigRanking.Builder ranking) {
            this.ranking = ranking.build();
            return this;
        }

        public Builder topK(Integer topK) {
            this.topK = topK;
            return this;
        }

        public RagRetrievalConfig build() {
            return new RagRetrievalConfig(this);
        }
    }
}
