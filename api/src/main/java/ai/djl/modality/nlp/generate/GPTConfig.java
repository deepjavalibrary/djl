/*
 * Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.modality.nlp.generate;

/** GPTConfig is used to store the GPT parameters used to select different versions of GPT. */
public class GPTConfig {
    private int numAttentionHeads;
    private int numLayers;
    private long kvDim;

    public GPTConfig() {
        numAttentionHeads = 12;
        numLayers = 12;
        kvDim = 64;
    }

    /**
     * Gets the value of the numAttentionHeads.
     *
     * @return the value of numAttentionHeads
     */
    public int getNumAttentionHeads() {
        return numAttentionHeads;
    }

    /**
     * Gets the value of the numLayers.
     *
     * @return the value of numLayers
     */
    public int getNumLayers() {
        return numLayers;
    }

    public void setNumLayers(int numLayers) {
        this.numLayers = numLayers;
    }

    /**
     * Gets the value of the kvDim.
     *
     * @return the value of kvDim
     */
    public long getKvDim() {
        return kvDim;
    }
}
