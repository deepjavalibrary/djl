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

public class SearchConfig {

    private int k;
    private float alpha;
    private int beam;
    private int maxSeqLength;
    private long padTokenId;
    private long eosTokenId;
    private boolean suffixPadding;

    /** Constructs a new ContrastiveSearchConfig object with default values. */
    public SearchConfig() {
        this.k = 4;
        this.alpha = 0.6f;
        this.beam = 3;
        this.maxSeqLength = 30;
        this.eosTokenId = 50256;
        this.padTokenId = 50256;
        this.suffixPadding = false;
    }

    /**
     * Gets the value of the k.
     *
     * @return the value of k
     */
    public int getK() {
        return k;
    }

    public void setK(int k) {
        this.k = k;
    }

    /**
     * Gets the value of the alpha.
     *
     * @return the value of alpha
     */
    public float getAlpha() {
        return alpha;
    }

    public void setAlpha(float alpha) {
        this.alpha = alpha;
    }

    /**
     * Gets the value of the beam.
     *
     * @return the value of beam
     */
    public int getBeam() {
        return beam;
    }

    public void setBeam(int beam) {
        this.beam = beam;
    }

    /**
     * Gets the value of the maxSeqLength.
     *
     * @return the value of maxSeqLength
     */
    public int getMaxSeqLength() {
        return maxSeqLength;
    }

    public void setMaxSeqLength(int maxSeqLength) {
        this.maxSeqLength = maxSeqLength;
    }

    /**
     * Gets the value of the padTokenId.
     *
     * @return the value of padTokenId
     */
    public long getPadTokenId() {
        return padTokenId;
    }

    public void setPadTokenId(long padTokenId) {
        this.padTokenId = padTokenId;
    }

    /**
     * Gets the value of the eosTokenId.
     *
     * @return the value of eosTokenId
     */
    public long getEosTokenId() {
        return eosTokenId;
    }

    /**
     * Gets the value of the suffixPadding.
     *
     * @return the value of suffixPadding
     */
    public boolean isSuffixPadding() {
        return suffixPadding;
    }

    public void setSuffixPadding(boolean suffixPadding) {
        this.suffixPadding = suffixPadding;
    }
}
