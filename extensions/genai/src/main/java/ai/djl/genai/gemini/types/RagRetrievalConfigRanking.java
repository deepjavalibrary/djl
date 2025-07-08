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
public class RagRetrievalConfigRanking {

    private RagRetrievalConfigRankingLlmRanker llmRanker;
    private RagRetrievalConfigRankingRankService rankService;

    RagRetrievalConfigRanking(Builder builder) {
        llmRanker = builder.llmRanker;
        rankService = builder.rankService;
    }

    public RagRetrievalConfigRankingLlmRanker getLlmRanker() {
        return llmRanker;
    }

    public RagRetrievalConfigRankingRankService getRankService() {
        return rankService;
    }

    public static Builder builder() {
        return new Builder();
    }

    /** Builder class for {@code RagRetrievalConfigRanking}. */
    public static final class Builder {

        RagRetrievalConfigRankingLlmRanker llmRanker;
        RagRetrievalConfigRankingRankService rankService;

        public Builder llmRanker(RagRetrievalConfigRankingLlmRanker llmRanker) {
            this.llmRanker = llmRanker;
            return this;
        }

        public Builder llmRanker(RagRetrievalConfigRankingLlmRanker.Builder llmRanker) {
            this.llmRanker = llmRanker.build();
            return this;
        }

        public Builder rankService(RagRetrievalConfigRankingRankService rankService) {
            this.rankService = rankService;
            return this;
        }

        public Builder rankService(RagRetrievalConfigRankingRankService.Builder rankService) {
            this.rankService = rankService.build();
            return this;
        }

        public RagRetrievalConfigRanking build() {
            return new RagRetrievalConfigRanking(this);
        }
    }
}
