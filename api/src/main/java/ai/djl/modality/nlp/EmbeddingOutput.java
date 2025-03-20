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
package ai.djl.modality.nlp;

import com.google.gson.annotations.SerializedName;

import java.util.LinkedHashMap;
import java.util.Map;

/** A class represents text embedding output. */
public class EmbeddingOutput {

    @SerializedName("dense_vecs")
    private float[] denseEmbedding;

    @SerializedName("lexical_weights")
    private LinkedHashMap<String, Float> lexicalWeights; // NOPMD

    /**
     * Returns the dense embedding.
     *
     * @return the dense embedding
     */
    public float[] getDenseEmbedding() {
        return denseEmbedding;
    }

    /**
     * Sets the dense embedding.
     *
     * @param denseEmbedding the dense embedding
     */
    public void setDenseEmbedding(float[] denseEmbedding) {
        this.denseEmbedding = denseEmbedding;
    }

    /**
     * Returns the token weights map.
     *
     * @return the token weights map
     */
    public Map<String, Float> getLexicalWeights() {
        return lexicalWeights;
    }

    /**
     * Adds the token weights to the output.
     *
     * @param tokenId the token id
     * @param weights the token weights
     */
    public void addTokenWeights(String tokenId, float weights) {
        if (lexicalWeights == null) {
            lexicalWeights = new LinkedHashMap<>(); // NOPMD
        }
        lexicalWeights.put(tokenId, weights);
    }
}
