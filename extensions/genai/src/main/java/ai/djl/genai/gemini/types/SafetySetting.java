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
public class SafetySetting {

    private HarmCategory category;
    private HarmBlockMethod method;
    private HarmBlockThreshold threshold;

    SafetySetting(Builder builder) {
        category = builder.category;
        method = builder.method;
        threshold = builder.threshold;
    }

    public HarmCategory getCategory() {
        return category;
    }

    public HarmBlockMethod getMethod() {
        return method;
    }

    public HarmBlockThreshold getThreshold() {
        return threshold;
    }

    public static Builder builder() {
        return new Builder();
    }

    /** Builder class for {@code SafetySetting}. */
    public static final class Builder {

        HarmCategory category;
        HarmBlockMethod method;
        HarmBlockThreshold threshold;

        public Builder category(HarmCategory category) {
            this.category = category;
            return this;
        }

        public Builder method(HarmBlockMethod method) {
            this.method = method;
            return this;
        }

        public Builder threshold(HarmBlockThreshold threshold) {
            this.threshold = threshold;
            return this;
        }

        public SafetySetting build() {
            return new SafetySetting(this);
        }
    }
}
