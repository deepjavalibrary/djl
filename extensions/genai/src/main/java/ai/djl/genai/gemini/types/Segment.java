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
public class Segment {

    private Integer endIndex;
    private Integer partIndex;
    private Integer startIndex;
    private String text;

    Segment(Builder builder) {
        endIndex = builder.endIndex;
        partIndex = builder.partIndex;
        startIndex = builder.startIndex;
        text = builder.text;
    }

    public Integer getEndIndex() {
        return endIndex;
    }

    public Integer getPartIndex() {
        return partIndex;
    }

    public Integer getStartIndex() {
        return startIndex;
    }

    public String getText() {
        return text;
    }

    public static Builder builder() {
        return new Builder();
    }

    /** Builder class for {@code Segment}. */
    public static final class Builder {

        Integer endIndex;
        Integer partIndex;
        Integer startIndex;
        String text;

        public Builder endIndex(Integer endIndex) {
            this.endIndex = endIndex;
            return this;
        }

        public Builder partIndex(Integer partIndex) {
            this.partIndex = partIndex;
            return this;
        }

        public Builder startIndex(Integer startIndex) {
            this.startIndex = startIndex;
            return this;
        }

        public Builder text(String text) {
            this.text = text;
            return this;
        }

        public Segment build() {
            return new Segment(this);
        }
    }
}
