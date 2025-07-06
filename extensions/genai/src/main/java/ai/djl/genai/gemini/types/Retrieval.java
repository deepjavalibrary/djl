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
public class Retrieval {

    private Boolean disableAttribution;
    private VertexRagStore vertexRagStore;

    Retrieval(Builder builder) {
        disableAttribution = builder.disableAttribution;
        vertexRagStore = builder.vertexRagStore;
    }

    public Boolean getDisableAttribution() {
        return disableAttribution;
    }

    public VertexRagStore getVertexRagStore() {
        return vertexRagStore;
    }

    public static Builder builder() {
        return new Builder();
    }

    /** Builder class for {@code Retrieval}. */
    public static final class Builder {

        Boolean disableAttribution;
        VertexRagStore vertexRagStore;

        public Builder disableAttribution(Boolean disableAttribution) {
            this.disableAttribution = disableAttribution;
            return this;
        }

        public Builder vertexRagStore(VertexRagStore vertexRagStore) {
            this.vertexRagStore = vertexRagStore;
            return this;
        }

        public Builder vertexRagStore(VertexRagStore.Builder vertexRagStore) {
            this.vertexRagStore = vertexRagStore.build();
            return this;
        }

        public Retrieval build() {
            return new Retrieval(this);
        }
    }
}
