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
public class DynamicRetrievalConfig {

    private Float dynamicThreshold;
    private DynamicRetrievalConfigMode mode;

    DynamicRetrievalConfig(Builder builder) {
        dynamicThreshold = builder.dynamicThreshold;
        mode = builder.mode;
    }

    public Float getDynamicThreshold() {
        return dynamicThreshold;
    }

    public DynamicRetrievalConfigMode getMode() {
        return mode;
    }

    public static Builder builder() {
        return new Builder();
    }

    /** Builder class for {@code DynamicRetrievalConfig}. */
    public static final class Builder {

        Float dynamicThreshold;
        DynamicRetrievalConfigMode mode;

        public Builder dynamicThreshold(Float dynamicThreshold) {
            this.dynamicThreshold = dynamicThreshold;
            return this;
        }

        public Builder mode(DynamicRetrievalConfigMode mode) {
            this.mode = mode;
            return this;
        }

        public DynamicRetrievalConfig build() {
            return new DynamicRetrievalConfig(this);
        }
    }
}
