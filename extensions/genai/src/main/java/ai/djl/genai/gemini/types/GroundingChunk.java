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
public class GroundingChunk {

    private GroundingChunkRetrievedContext retrievedContext;
    private GroundingChunkWeb web;

    GroundingChunk(Builder builder) {
        retrievedContext = builder.retrievedContext;
        web = builder.web;
    }

    public GroundingChunkRetrievedContext getRetrievedContext() {
        return retrievedContext;
    }

    public GroundingChunkWeb getWeb() {
        return web;
    }

    public static Builder builder() {
        return new Builder();
    }

    /** Builder class for {@code GroundingChunk}. */
    public static final class Builder {

        GroundingChunkRetrievedContext retrievedContext;
        GroundingChunkWeb web;

        public Builder retrievedContext(GroundingChunkRetrievedContext retrievedContext) {
            this.retrievedContext = retrievedContext;
            return this;
        }

        public Builder retrievedContext(GroundingChunkRetrievedContext.Builder retrievedContext) {
            this.retrievedContext = retrievedContext.build();
            return this;
        }

        public Builder web(GroundingChunkWeb web) {
            this.web = web;
            return this;
        }

        public Builder web(GroundingChunkWeb.Builder web) {
            this.web = web.build();
            return this;
        }

        public GroundingChunk build() {
            return new GroundingChunk(this);
        }
    }
}
