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
public class LogprobsResultCandidate {

    private Float logProbability;
    private String token;
    private Integer tokenId;

    LogprobsResultCandidate(Builder builder) {
        logProbability = builder.logProbability;
        token = builder.token;
        tokenId = builder.tokenId;
    }

    public Float getLogProbability() {
        return logProbability;
    }

    public String getToken() {
        return token;
    }

    public Integer getTokenId() {
        return tokenId;
    }

    public static Builder builder() {
        return new Builder();
    }

    /** Builder class for {@code LogprobsResultCandidate}. */
    public static final class Builder {

        Float logProbability;
        String token;
        Integer tokenId;

        public Builder logProbability(Float logProbability) {
            this.logProbability = logProbability;
            return this;
        }

        public Builder token(String token) {
            this.token = token;
            return this;
        }

        public Builder tokenId(Integer tokenId) {
            this.tokenId = tokenId;
            return this;
        }

        public LogprobsResultCandidate build() {
            return new LogprobsResultCandidate(this);
        }
    }
}
