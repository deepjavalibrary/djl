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
public class LogprobsResultTopCandidates {

    private List<LogprobsResultCandidate> candidates;

    LogprobsResultTopCandidates(Builder builder) {
        candidates = builder.candidates;
    }

    public List<LogprobsResultCandidate> getCandidates() {
        return candidates;
    }

    public static Builder builder() {
        return new Builder();
    }

    /** Builder class for {@code LogprobsResultTopCandidates}. */
    public static final class Builder {

        List<LogprobsResultCandidate> candidates = new ArrayList<>();

        public Builder candidates(List<LogprobsResultCandidate> candidates) {
            this.candidates.clear();
            this.candidates.addAll(candidates);
            return this;
        }

        public Builder addCandidate(LogprobsResultCandidate candidate) {
            this.candidates.add(candidate);
            return this;
        }

        public Builder addCandidate(LogprobsResultCandidate.Builder candidate) {
            this.candidates.add(candidate.build());
            return this;
        }

        public LogprobsResultTopCandidates build() {
            return new LogprobsResultTopCandidates(this);
        }
    }
}
