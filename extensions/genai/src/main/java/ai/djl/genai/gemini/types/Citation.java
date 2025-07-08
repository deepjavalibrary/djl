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
public class Citation {

    private Integer endIndex;
    private String license;
    private GoogleTypeDate publicationDate;
    private Integer startIndex;
    private String title;
    private String uri;

    Citation(Builder builder) {
        endIndex = builder.endIndex;
        license = builder.license;
        publicationDate = builder.publicationDate;
        startIndex = builder.startIndex;
        title = builder.title;
        uri = builder.uri;
    }

    public Integer getEndIndex() {
        return endIndex;
    }

    public String getLicense() {
        return license;
    }

    public GoogleTypeDate getPublicationDate() {
        return publicationDate;
    }

    public Integer getStartIndex() {
        return startIndex;
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

    /** Builder class for {@code Citation}. */
    public static final class Builder {

        Integer endIndex;
        String license;
        GoogleTypeDate publicationDate;
        Integer startIndex;
        String title;
        String uri;

        public Builder endIndex(Integer endIndex) {
            this.endIndex = endIndex;
            return this;
        }

        public Builder license(String license) {
            this.license = license;
            return this;
        }

        public Builder publicationDate(GoogleTypeDate publicationDate) {
            this.publicationDate = publicationDate;
            return this;
        }

        public Builder publicationDate(GoogleTypeDate.Builder publicationDate) {
            this.publicationDate = publicationDate.build();
            return this;
        }

        public Builder startIndex(Integer startIndex) {
            this.startIndex = startIndex;
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

        public Citation build() {
            return new Citation(this);
        }
    }
}
