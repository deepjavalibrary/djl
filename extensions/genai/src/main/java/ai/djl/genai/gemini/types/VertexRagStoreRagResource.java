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

import java.util.List;

/** A data class represents Gemini schema. */
@SuppressWarnings("MissingJavadocMethod")
public class VertexRagStoreRagResource {

    private String ragCorpus;
    private List<String> ragFileIds;

    VertexRagStoreRagResource(Builder builder) {
        ragCorpus = builder.ragCorpus;
        ragFileIds = builder.ragFileIds;
    }

    public String getRagCorpus() {
        return ragCorpus;
    }

    public List<String> getRagFileIds() {
        return ragFileIds;
    }

    public static Builder builder() {
        return new Builder();
    }

    /** Builder class for {@code VertexRagStoreRagResource}. */
    public static final class Builder {

        String ragCorpus;
        List<String> ragFileIds;

        public Builder ragCorpus(String ragCorpus) {
            this.ragCorpus = ragCorpus;
            return this;
        }

        public Builder ragFileIds(List<String> ragFileIds) {
            this.ragFileIds = ragFileIds;
            return this;
        }

        public VertexRagStoreRagResource build() {
            return new VertexRagStoreRagResource(this);
        }
    }
}
