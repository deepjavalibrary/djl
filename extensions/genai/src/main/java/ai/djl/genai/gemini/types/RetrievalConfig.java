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
public class RetrievalConfig {

    private String languageCode;
    private LatLng latLng;

    RetrievalConfig(Builder builder) {
        languageCode = builder.languageCode;
        latLng = builder.latLng;
    }

    public String getLanguageCode() {
        return languageCode;
    }

    public LatLng getLatLng() {
        return latLng;
    }

    public static Builder builder() {
        return new Builder();
    }

    /** Builder class for {@code RetrievalConfig}. */
    public static final class Builder {

        String languageCode;
        LatLng latLng;

        public Builder languageCode(String languageCode) {
            this.languageCode = languageCode;
            return this;
        }

        public Builder latLng(LatLng latLng) {
            this.latLng = latLng;
            return this;
        }

        public RetrievalConfig build() {
            return new RetrievalConfig(this);
        }
    }
}
