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
public class UrlMetadata {

    private String retrievedUrl;
    private UrlRetrievalStatus urlRetrievalStatus;

    UrlMetadata(Builder builder) {
        retrievedUrl = builder.retrievedUrl;
        urlRetrievalStatus = builder.urlRetrievalStatus;
    }

    public String getRetrievedUrl() {
        return retrievedUrl;
    }

    public UrlRetrievalStatus getUrlRetrievalStatus() {
        return urlRetrievalStatus;
    }

    public static Builder builder() {
        return new Builder();
    }

    /** Builder class for {@code UrlMetadata}. */
    public static final class Builder {

        String retrievedUrl;
        UrlRetrievalStatus urlRetrievalStatus;

        public Builder retrievedUrl(String retrievedUrl) {
            this.retrievedUrl = retrievedUrl;
            return this;
        }

        public Builder urlRetrievalStatus(UrlRetrievalStatus urlRetrievalStatus) {
            this.urlRetrievalStatus = urlRetrievalStatus;
            return this;
        }

        public UrlMetadata build() {
            return new UrlMetadata(this);
        }
    }
}
