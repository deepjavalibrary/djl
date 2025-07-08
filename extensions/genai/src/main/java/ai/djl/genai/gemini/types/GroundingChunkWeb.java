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
public class GroundingChunkWeb {

    private String domain;
    private String title;
    private String uri;

    GroundingChunkWeb(Builder builder) {
        domain = builder.domain;
        title = builder.title;
        uri = builder.uri;
    }

    public String getDomain() {
        return domain;
    }

    public String getTitle() {
        return title;
    }

    public String getUri() {
        return uri;
    }

    public static Builder builder() {
        return new Builder();
    }

    /** Builder class for {@code GroundingChunkWeb}. */
    public static final class Builder {

        String domain;
        String title;
        String uri;

        public Builder domain(String domain) {
            this.domain = domain;
            return this;
        }

        public Builder title(String title) {
            this.title = title;
            return this;
        }

        public Builder uri(String uri) {
            this.uri = uri;
            return this;
        }

        public GroundingChunkWeb build() {
            return new GroundingChunkWeb(this);
        }
    }
}
