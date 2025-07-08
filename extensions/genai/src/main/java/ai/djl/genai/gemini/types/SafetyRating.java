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
public class SafetyRating {

    private Boolean blocked;
    private HarmCategory category;
    private HarmBlockThreshold overwrittenThreshold;
    private HarmProbability probability;
    private Float probabilityScore;
    private HarmSeverity severity;
    private Float severityScore;

    SafetyRating(Builder builder) {
        blocked = builder.blocked;
        category = builder.category;
        overwrittenThreshold = builder.overwrittenThreshold;
        probability = builder.probability;
        probabilityScore = builder.probabilityScore;
        severity = builder.severity;
        severityScore = builder.severityScore;
    }

    public Boolean getBlocked() {
        return blocked;
    }

    public HarmCategory getCategory() {
        return category;
    }

    public HarmBlockThreshold getOverwrittenThreshold() {
        return overwrittenThreshold;
    }

    public HarmProbability getProbability() {
        return probability;
    }

    public Float getProbabilityScore() {
        return probabilityScore;
    }

    public HarmSeverity getSeverity() {
        return severity;
    }

    public Float getSeverityScore() {
        return severityScore;
    }

    public static Builder builder() {
        return new Builder();
    }

    /** Builder class for {@code SafetyRating}. */
    public static final class Builder {

        Boolean blocked;
        HarmCategory category;
        HarmBlockThreshold overwrittenThreshold;
        HarmProbability probability;
        Float probabilityScore;
        HarmSeverity severity;
        Float severityScore;

        public Builder blocked(Boolean blocked) {
            this.blocked = blocked;
            return this;
        }

        public Builder category(HarmCategory category) {
            this.category = category;
            return this;
        }

        public Builder overwrittenThreshold(HarmBlockThreshold overwrittenThreshold) {
            this.overwrittenThreshold = overwrittenThreshold;
            return this;
        }

        public Builder probability(HarmProbability probability) {
            this.probability = probability;
            return this;
        }

        public Builder probabilityScore(Float probabilityScore) {
            this.probabilityScore = probabilityScore;
            return this;
        }

        public Builder severity(HarmSeverity severity) {
            this.severity = severity;
            return this;
        }

        public Builder severityScore(Float severityScore) {
            this.severityScore = severityScore;
            return this;
        }

        public SafetyRating build() {
            return new SafetyRating(this);
        }
    }
}
