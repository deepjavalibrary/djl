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
public class CitationMetadata {

    private List<Citation> citations;

    CitationMetadata(Builder builder) {
        citations = builder.citations;
    }

    public List<Citation> getCitations() {
        return citations;
    }

    public static Builder builder() {
        return new Builder();
    }

    /** Builder class for {@code CitationMetadata}. */
    public static final class Builder {

        List<Citation> citations = new ArrayList<>();

        public Builder citations(List<Citation> citations) {
            this.citations.clear();
            this.citations.addAll(citations);
            return this;
        }

        public Builder addCitation(Citation citation) {
            this.citations.add(citation);
            return this;
        }

        public Builder addCitation(Citation.Builder citation) {
            this.citations.add(citation.build());
            return this;
        }

        public CitationMetadata build() {
            return new CitationMetadata(this);
        }
    }
}
