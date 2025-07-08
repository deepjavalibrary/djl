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
public class UrlContextMetadata {

    private List<UrlMetadata> urlMetadata;

    UrlContextMetadata(Builder builder) {
        urlMetadata = builder.urlMetadata;
    }

    public List<UrlMetadata> getUrlMetadata() {
        return urlMetadata;
    }

    public static Builder builder() {
        return new Builder();
    }

    /** Builder class for {@code UrlContextMetadata}. */
    public static final class Builder {

        List<UrlMetadata> urlMetadata = new ArrayList<>();

        public Builder urlMetadata(List<UrlMetadata> urlMetadata) {
            this.urlMetadata.clear();
            this.urlMetadata.addAll(urlMetadata);
            return this;
        }

        public Builder addUrlMetadata(UrlMetadata urlMetadata) {
            this.urlMetadata.add(urlMetadata);
            return this;
        }

        public Builder addUrlMetadata(UrlMetadata.Builder urlMetadata) {
            this.urlMetadata.add(urlMetadata.build());
            return this;
        }

        public UrlContextMetadata build() {
            return new UrlContextMetadata(this);
        }
    }
}
