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
public class GenerateContentResponse {

    private List<Content> automaticFunctionCallingHistory;
    private List<Candidate> candidates;
    private String createTime;
    private String modelVersion;
    private PromptFeedback promptFeedback;
    private String responseId;
    private UsageMetadata usageMetadata;

    GenerateContentResponse(Builder builder) {
        automaticFunctionCallingHistory = builder.automaticFunctionCallingHistory;
        candidates = builder.candidates;
        createTime = builder.createTime;
        modelVersion = builder.modelVersion;
        promptFeedback = builder.promptFeedback;
        responseId = builder.responseId;
        usageMetadata = builder.usageMetadata;
    }

    public List<Content> getAutomaticFunctionCallingHistory() {
        return automaticFunctionCallingHistory;
    }

    public List<Candidate> getCandidates() {
        return candidates;
    }

    public String getCreateTime() {
        return createTime;
    }

    public String getModelVersion() {
        return modelVersion;
    }

    public PromptFeedback getPromptFeedback() {
        return promptFeedback;
    }

    public String getResponseId() {
        return responseId;
    }

    public UsageMetadata getUsageMetadata() {
        return usageMetadata;
    }

    public static Builder builder() {
        return new Builder();
    }

    /** Builder class for {@code GenerateContentResponse}. */
    public static final class Builder {

        List<Content> automaticFunctionCallingHistory = new ArrayList<>();
        List<Candidate> candidates = new ArrayList<>();
        String createTime;
        String modelVersion;
        PromptFeedback promptFeedback;
        String responseId;
        UsageMetadata usageMetadata;

        public Builder automaticFunctionCallingHistory(
                List<Content> automaticFunctionCallingHistory) {
            this.automaticFunctionCallingHistory.clear();
            this.automaticFunctionCallingHistory.addAll(automaticFunctionCallingHistory);
            return this;
        }

        public Builder addAutomaticFunctionCallingHistory(Content automaticFunctionCallingHistory) {
            this.automaticFunctionCallingHistory.add(automaticFunctionCallingHistory);
            return this;
        }

        public Builder addAutomaticFunctionCallingHistory(
                Content.Builder automaticFunctionCallingHistory) {
            this.automaticFunctionCallingHistory.add(automaticFunctionCallingHistory.build());
            return this;
        }

        public Builder candidates(List<Candidate> candidates) {
            this.candidates.clear();
            this.candidates.addAll(candidates);
            return this;
        }

        public Builder addCandidate(Candidate candidate) {
            this.candidates.add(candidate);
            return this;
        }

        public Builder addCandidate(Candidate.Builder candidate) {
            this.candidates.add(candidate.build());
            return this;
        }

        public Builder createTime(String createTime) {
            this.createTime = createTime;
            return this;
        }

        public Builder modelVersion(String modelVersion) {
            this.modelVersion = modelVersion;
            return this;
        }

        public Builder promptFeedback(PromptFeedback promptFeedback) {
            this.promptFeedback = promptFeedback;
            return this;
        }

        public Builder promptFeedback(PromptFeedback.Builder promptFeedback) {
            this.promptFeedback = promptFeedback.build();
            return this;
        }

        public Builder responseId(String responseId) {
            this.responseId = responseId;
            return this;
        }

        public Builder usageMetadata(UsageMetadata usageMetadata) {
            this.usageMetadata = usageMetadata;
            return this;
        }

        public Builder usageMetadata(UsageMetadata.Builder usageMetadata) {
            this.usageMetadata = usageMetadata.build();
            return this;
        }

        public GenerateContentResponse build() {
            return new GenerateContentResponse(this);
        }
    }
}
