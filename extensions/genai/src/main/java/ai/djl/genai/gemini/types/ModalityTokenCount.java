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
public class ModalityTokenCount {

    private MediaModality modality;
    private Integer tokenCount;

    ModalityTokenCount(Builder builder) {
        modality = builder.modality;
        tokenCount = builder.tokenCount;
    }

    public MediaModality getModality() {
        return modality;
    }

    public Integer getTokenCount() {
        return tokenCount;
    }

    public static Builder builder() {
        return new Builder();
    }

    /** Builder class for {@code ModalityTokenCount}. */
    public static final class Builder {

        MediaModality modality;
        Integer tokenCount;

        public Builder modality(MediaModality modality) {
            this.modality = modality;
            return this;
        }

        public Builder tokenCount(Integer tokenCount) {
            this.tokenCount = tokenCount;
            return this;
        }

        public ModalityTokenCount build() {
            return new ModalityTokenCount(this);
        }
    }
}
