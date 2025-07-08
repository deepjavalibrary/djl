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
public class LogprobsResult {

    private List<LogprobsResultCandidate> chosenCandidates;
    private List<LogprobsResultTopCandidates> topCandidates;

    LogprobsResult(Builder builder) {
        chosenCandidates = builder.chosenCandidates;
        topCandidates = builder.topCandidates;
    }

    public List<LogprobsResultCandidate> getChosenCandidates() {
        return chosenCandidates;
    }

    public List<LogprobsResultTopCandidates> getTopCandidates() {
        return topCandidates;
    }

    public static Builder builder() {
        return new Builder();
    }

    /** Builder class for {@code LogprobsResult}. */
    public static final class Builder {

        List<LogprobsResultCandidate> chosenCandidates = new ArrayList<>();
        List<LogprobsResultTopCandidates> topCandidates = new ArrayList<>();

        public Builder chosenCandidates(List<LogprobsResultCandidate> chosenCandidates) {
            this.chosenCandidates.clear();
            this.chosenCandidates.addAll(chosenCandidates);
            return this;
        }

        public Builder addChosenCandidate(LogprobsResultCandidate chosenCandidate) {
            this.chosenCandidates.add(chosenCandidate);
            return this;
        }

        public Builder addChosenCandidate(LogprobsResultCandidate.Builder chosenCandidate) {
            this.chosenCandidates.add(chosenCandidate.build());
            return this;
        }

        public Builder topCandidates(List<LogprobsResultTopCandidates> topCandidates) {
            this.topCandidates.clear();
            this.topCandidates.addAll(topCandidates);
            return this;
        }

        public Builder addTopCandidate(LogprobsResultTopCandidates topCandidate) {
            this.topCandidates.add(topCandidate);
            return this;
        }

        public Builder addTopCandidate(LogprobsResultTopCandidates.Builder topCandidate) {
            this.topCandidates.add(topCandidate.build());
            return this;
        }

        public LogprobsResult build() {
            return new LogprobsResult(this);
        }
    }
}
