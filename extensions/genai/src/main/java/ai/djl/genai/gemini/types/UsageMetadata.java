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
public class UsageMetadata {

    private List<ModalityTokenCount> cacheTokensDetails;
    private Integer cachedContentTokenCount;
    private Integer candidatesTokenCount;
    private List<ModalityTokenCount> candidatesTokensDetails;
    private Integer promptTokenCount;
    private List<ModalityTokenCount> promptTokensDetails;
    private Integer thoughtsTokenCount;
    private Integer toolUsePromptTokenCount;
    private List<ModalityTokenCount> toolUsePromptTokensDetails;
    private Integer totalTokenCount;
    private TrafficType trafficType;

    UsageMetadata(Builder builder) {
        cacheTokensDetails = builder.cacheTokensDetails;
        cachedContentTokenCount = builder.cachedContentTokenCount;
        candidatesTokenCount = builder.candidatesTokenCount;
        candidatesTokensDetails = builder.candidatesTokensDetails;
        promptTokenCount = builder.promptTokenCount;
        promptTokensDetails = builder.promptTokensDetails;
        thoughtsTokenCount = builder.thoughtsTokenCount;
        toolUsePromptTokenCount = builder.toolUsePromptTokenCount;
        toolUsePromptTokensDetails = builder.toolUsePromptTokensDetails;
        totalTokenCount = builder.totalTokenCount;
        trafficType = builder.trafficType;
    }

    public List<ModalityTokenCount> getCacheTokensDetails() {
        return cacheTokensDetails;
    }

    public Integer getCachedContentTokenCount() {
        return cachedContentTokenCount;
    }

    public Integer getCandidatesTokenCount() {
        return candidatesTokenCount;
    }

    public List<ModalityTokenCount> getCandidatesTokensDetails() {
        return candidatesTokensDetails;
    }

    public Integer getPromptTokenCount() {
        return promptTokenCount;
    }

    public List<ModalityTokenCount> getPromptTokensDetails() {
        return promptTokensDetails;
    }

    public Integer getThoughtsTokenCount() {
        return thoughtsTokenCount;
    }

    public Integer getToolUsePromptTokenCount() {
        return toolUsePromptTokenCount;
    }

    public List<ModalityTokenCount> getToolUsePromptTokensDetails() {
        return toolUsePromptTokensDetails;
    }

    public Integer getTotalTokenCount() {
        return totalTokenCount;
    }

    public TrafficType getTrafficType() {
        return trafficType;
    }

    public static Builder builder() {
        return new Builder();
    }

    /** Builder class for {@code UsageMetadata}. */
    public static final class Builder {

        List<ModalityTokenCount> cacheTokensDetails = new ArrayList<>();
        Integer cachedContentTokenCount;
        Integer candidatesTokenCount;
        List<ModalityTokenCount> candidatesTokensDetails = new ArrayList<>();
        Integer promptTokenCount;
        List<ModalityTokenCount> promptTokensDetails = new ArrayList<>();
        Integer thoughtsTokenCount;
        Integer toolUsePromptTokenCount;
        List<ModalityTokenCount> toolUsePromptTokensDetails = new ArrayList<>();
        Integer totalTokenCount;
        TrafficType trafficType;

        public Builder cacheTokensDetails(List<ModalityTokenCount> cacheTokensDetails) {
            this.cacheTokensDetails.clear();
            this.cacheTokensDetails.addAll(cacheTokensDetails);
            return this;
        }

        public Builder addCacheTokensDetail(ModalityTokenCount cacheTokensDetail) {
            this.cacheTokensDetails.add(cacheTokensDetail);
            return this;
        }

        public Builder addCacheTokensDetail(ModalityTokenCount.Builder cacheTokensDetail) {
            this.cacheTokensDetails.add(cacheTokensDetail.build());
            return this;
        }

        public Builder cachedContentTokenCount(Integer cachedContentTokenCount) {
            this.cachedContentTokenCount = cachedContentTokenCount;
            return this;
        }

        public Builder candidatesTokenCount(Integer candidatesTokenCount) {
            this.candidatesTokenCount = candidatesTokenCount;
            return this;
        }

        public Builder candidatesTokensDetails(List<ModalityTokenCount> candidatesTokensDetails) {
            this.candidatesTokensDetails.clear();
            this.candidatesTokensDetails.addAll(candidatesTokensDetails);
            return this;
        }

        public Builder addCandidatesTokensDetail(ModalityTokenCount candidatesTokensDetail) {
            this.candidatesTokensDetails.add(candidatesTokensDetail);
            return this;
        }

        public Builder addCandidatesTokensDetail(
                ModalityTokenCount.Builder candidatesTokensDetail) {
            this.candidatesTokensDetails.add(candidatesTokensDetail.build());
            return this;
        }

        public Builder promptTokenCount(Integer promptTokenCount) {
            this.promptTokenCount = promptTokenCount;
            return this;
        }

        public Builder promptTokensDetails(List<ModalityTokenCount> promptTokensDetails) {
            this.promptTokensDetails.clear();
            this.promptTokensDetails.addAll(promptTokensDetails);
            return this;
        }

        public Builder addPromptTokensDetail(ModalityTokenCount promptTokensDetail) {
            this.promptTokensDetails.add(promptTokensDetail);
            return this;
        }

        public Builder addPromptTokensDetail(ModalityTokenCount.Builder promptTokensDetail) {
            this.promptTokensDetails.add(promptTokensDetail.build());
            return this;
        }

        public Builder thoughtsTokenCount(Integer thoughtsTokenCount) {
            this.thoughtsTokenCount = thoughtsTokenCount;
            return this;
        }

        public Builder toolUsePromptTokenCount(Integer toolUsePromptTokenCount) {
            this.toolUsePromptTokenCount = toolUsePromptTokenCount;
            return this;
        }

        public Builder toolUsePromptTokensDetails(
                List<ModalityTokenCount> toolUsePromptTokensDetails) {
            this.toolUsePromptTokensDetails.clear();
            this.toolUsePromptTokensDetails.addAll(toolUsePromptTokensDetails);
            return this;
        }

        public Builder addToolUsePromptTokensDetail(ModalityTokenCount toolUsePromptTokensDetail) {
            this.toolUsePromptTokensDetails.add(toolUsePromptTokensDetail);
            return this;
        }

        public Builder addToolUsePromptTokensDetail(
                ModalityTokenCount.Builder toolUsePromptTokensDetail) {
            this.toolUsePromptTokensDetails.add(toolUsePromptTokensDetail.build());
            return this;
        }

        public Builder totalTokenCount(Integer totalTokenCount) {
            this.totalTokenCount = totalTokenCount;
            return this;
        }

        public Builder trafficType(TrafficType trafficType) {
            this.trafficType = trafficType;
            return this;
        }

        public UsageMetadata build() {
            return new UsageMetadata(this);
        }
    }
}
