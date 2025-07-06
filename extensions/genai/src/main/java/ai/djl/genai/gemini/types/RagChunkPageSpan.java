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
public class RagChunkPageSpan {

    private Integer firstPage;
    private Integer lastPage;

    RagChunkPageSpan(Builder builder) {
        firstPage = builder.firstPage;
        lastPage = builder.lastPage;
    }

    public Integer getFirstPage() {
        return firstPage;
    }

    public Integer getLastPage() {
        return lastPage;
    }

    public static Builder builder() {
        return new Builder();
    }

    /** Builder class for {@code RagChunkPageSpan}. */
    public static final class Builder {

        Integer firstPage;
        Integer lastPage;

        public Builder firstPage(Integer firstPage) {
            this.firstPage = firstPage;
            return this;
        }

        public Builder lastPage(Integer lastPage) {
            this.lastPage = lastPage;
            return this;
        }

        public RagChunkPageSpan build() {
            return new RagChunkPageSpan(this);
        }
    }
}
