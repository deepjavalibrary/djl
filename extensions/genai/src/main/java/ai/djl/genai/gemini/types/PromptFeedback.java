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
public class PromptFeedback {

    private BlockedReason blockReason;
    private String blockReasonMessage;
    private List<SafetyRating> safetyRatings;

    PromptFeedback(Builder builder) {
        blockReason = builder.blockReason;
        blockReasonMessage = builder.blockReasonMessage;
        safetyRatings = builder.safetyRatings;
    }

    public BlockedReason getBlockReason() {
        return blockReason;
    }

    public String getBlockReasonMessage() {
        return blockReasonMessage;
    }

    public List<SafetyRating> getSafetyRatings() {
        return safetyRatings;
    }

    public static Builder builder() {
        return new Builder();
    }

    /** Builder class for {@code PromptFeedback}. */
    public static final class Builder {

        BlockedReason blockReason;
        String blockReasonMessage;
        List<SafetyRating> safetyRatings = new ArrayList<>();

        public Builder blockReason(BlockedReason blockReason) {
            this.blockReason = blockReason;
            return this;
        }

        public Builder blockReasonMessage(String blockReasonMessage) {
            this.blockReasonMessage = blockReasonMessage;
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

        public PromptFeedback build() {
            return new PromptFeedback(this);
        }
    }
}
