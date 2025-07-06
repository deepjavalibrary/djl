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
public class Candidate {

    private Double avgLogprobs;
    private CitationMetadata citationMetadata;
    private Content content;
    private String finishMessage;
    private FinishReason finishReason;
    private GroundingMetadata groundingMetadata;
    private Integer index;
    private LogprobsResult logprobsResult;
    private List<SafetyRating> safetyRatings;
    private Integer tokenCount;
    private UrlContextMetadata urlContextMetadata;

    Candidate(Builder builder) {
        avgLogprobs = builder.avgLogprobs;
        citationMetadata = builder.citationMetadata;
        content = builder.content;
        finishMessage = builder.finishMessage;
        finishReason = builder.finishReason;
        groundingMetadata = builder.groundingMetadata;
        index = builder.index;
        logprobsResult = builder.logprobsResult;
        safetyRatings = builder.safetyRatings;
        tokenCount = builder.tokenCount;
        urlContextMetadata = builder.urlContextMetadata;
    }

    public Double getAvgLogprobs() {
        return avgLogprobs;
    }

    public CitationMetadata getCitationMetadata() {
        return citationMetadata;
    }

    public Content getContent() {
        return content;
    }

    public String getFinishMessage() {
        return finishMessage;
    }

    public FinishReason getFinishReason() {
        return finishReason;
    }

    public GroundingMetadata getGroundingMetadata() {
        return groundingMetadata;
    }

    public Integer getIndex() {
        return index;
    }

    public LogprobsResult getLogprobsResult() {
        return logprobsResult;
    }

    public List<SafetyRating> getSafetyRatings() {
        return safetyRatings;
    }

    public Integer getTokenCount() {
        return tokenCount;
    }

    public UrlContextMetadata getUrlContextMetadata() {
        return urlContextMetadata;
    }

    public static Builder builder() {
        return new Builder();
    }

    /** Builder class for {@code Candidate}. */
    public static final class Builder {

        Double avgLogprobs;
        CitationMetadata citationMetadata;
        Content content;
        String finishMessage;
        FinishReason finishReason;
        GroundingMetadata groundingMetadata;
        Integer index;
        LogprobsResult logprobsResult;
        List<SafetyRating> safetyRatings = new ArrayList<>();
        Integer tokenCount;
        UrlContextMetadata urlContextMetadata;

        public Builder avgLogprobs(Double avgLogprobs) {
            this.avgLogprobs = avgLogprobs;
            return this;
        }

        public Builder citationMetadata(CitationMetadata citationMetadata) {
            this.citationMetadata = citationMetadata;
            return this;
        }

        public Builder citationMetadata(CitationMetadata.Builder citationMetadata) {
            this.citationMetadata = citationMetadata.build();
            return this;
        }

        public Builder content(Content content) {
            this.content = content;
            return this;
        }

        public Builder content(Content.Builder content) {
            this.content = content.build();
            return this;
        }

        public Builder finishMessage(String finishMessage) {
            this.finishMessage = finishMessage;
            return this;
        }

        public Builder finishReason(FinishReason finishReason) {
            this.finishReason = finishReason;
            return this;
        }

        public Builder groundingMetadata(GroundingMetadata groundingMetadata) {
            this.groundingMetadata = groundingMetadata;
            return this;
        }

        public Builder groundingMetadata(GroundingMetadata.Builder groundingMetadata) {
            this.groundingMetadata = groundingMetadata.build();
            return this;
        }

        public Builder index(Integer index) {
            this.index = index;
            return this;
        }

        public Builder logprobsResult(LogprobsResult logprobsResult) {
            this.logprobsResult = logprobsResult;
            return this;
        }

        public Builder logprobsResult(LogprobsResult.Builder logprobsResult) {
            this.logprobsResult = logprobsResult.build();
            return this;
        }

        public Builder safetyRatings(List<SafetyRating> safetyRatings) {
            this.safetyRatings.clear();
            this.safetyRatings.addAll(safetyRatings);
            return this;
        }

        public Builder addSafetyRating(SafetyRating safetyRating) {
            this.safetyRatings.add(safetyRating);
            return this;
        }

        public Builder addSafetyRating(SafetyRating.Builder safetyRating) {
            this.safetyRatings.add(safetyRating.build());
            return this;
        }

        public Builder tokenCount(Integer tokenCount) {
            this.tokenCount = tokenCount;
            return this;
        }

        public Builder urlContextMetadata(UrlContextMetadata urlContextMetadata) {
            this.urlContextMetadata = urlContextMetadata;
            return this;
        }

        public Builder urlContextMetadata(UrlContextMetadata.Builder urlContextMetadata) {
            this.urlContextMetadata = urlContextMetadata.build();
            return this;
        }

        public Candidate build() {
            return new Candidate(this);
        }
    }
}
