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
public class GoogleSearchRetrieval {

    private DynamicRetrievalConfig dynamicRetrievalConfig;

    GoogleSearchRetrieval(Builder builder) {
        dynamicRetrievalConfig = builder.dynamicRetrievalConfig;
    }

    public DynamicRetrievalConfig getDynamicRetrievalConfig() {
        return dynamicRetrievalConfig;
    }

    public static Builder builder() {
        return new Builder();
    }

    /** Builder class for {@code GoogleSearchRetrieval}. */
    public static final class Builder {

        DynamicRetrievalConfig dynamicRetrievalConfig;

        public Builder dynamicRetrievalConfig(DynamicRetrievalConfig dynamicRetrievalConfig) {
            this.dynamicRetrievalConfig = dynamicRetrievalConfig;
            return this;
        }

        public Builder dynamicRetrievalConfig(
                DynamicRetrievalConfig.Builder dynamicRetrievalConfig) {
            this.dynamicRetrievalConfig = dynamicRetrievalConfig.build();
            return this;
        }

        public GoogleSearchRetrieval build() {
            return new GoogleSearchRetrieval(this);
        }
    }
}
