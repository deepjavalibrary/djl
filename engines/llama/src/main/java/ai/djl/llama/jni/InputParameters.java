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
package ai.djl.llama.jni;

import java.util.Map;

/** A class holds input parameters. */
@SuppressWarnings("PMD.UnusedPrivateField")
public class InputParameters {

    private int nPredict = -1;
    private int nKeep;
    private int nProbs;
    private int topK = 40;
    private float topP = 0.95f;
    private float tfsZ = 1.00f;
    private float typicalP = 1.00f;
    private float temperature = 0.80f;
    private float repeatPenalty = 1.10f;
    private int repeatLastN = 64;
    private float frequencyPenalty;
    private float presencePenalty;
    private boolean penalizeNl;
    private boolean ignoreEos;
    private int mirostat;
    private float mirostatTau = 5.00f;
    private float mirostatEta = 0.10f;
    private int nBeams = 2;
    private int seed = 42;
    private Map<Integer, Float> logitBias;
    private String grammar;
    private String[] antiPrompt;

    /**
     * Returns the number of predict.
     *
     * @return the number of predict
     */
    public int getNumberPredict() {
        return nPredict;
    }

    /**
     * Returns the number of keep.
     *
     * @return the number of keep
     */
    public int getNumberKeep() {
        return nKeep;
    }

    /**
     * Returns the number of probabilities.
     *
     * @return the number of probabilities
     */
    public int getNumberProbabilities() {
        return nProbs;
    }

    /**
     * Returns the top K.
     *
     * @return the top K
     */
    public int getTopK() {
        return topK;
    }

    /**
     * Return the top P.
     *
     * @return the top P
     */
    public float getTopP() {
        return topP;
    }

    /**
     * Return the TfsZ.
     *
     * @return the TfsZ
     */
    public float getTfsZ() {
        return tfsZ;
    }

    /**
     * Return the typical P.
     *
     * @return the typical P
     */
    public float getTypicalP() {
        return typicalP;
    }

    /**
     * Return the temperature.
     *
     * @return the temperature
     */
    public float getTemperature() {
        return temperature;
    }

    /**
     * Return the repeat penalty.
     *
     * @return the repeat penalty
     */
    public float getRepeatPenalty() {
        return repeatPenalty;
    }

    /**
     * Return the repeat last N.
     *
     * @return the repeat last N
     */
    public int getRepeatLastN() {
        return repeatLastN;
    }

    /**
     * Return the frequency penalty.
     *
     * @return the frequency penalty
     */
    public float getFrequencyPenalty() {
        return frequencyPenalty;
    }

    /**
     * Return the presence penalty.
     *
     * @return the presence penalty
     */
    public float getPresencePenalty() {
        return presencePenalty;
    }

    /**
     * Return the penalize NL.
     *
     * @return the penalize NL
     */
    public boolean isPenalizeNl() {
        return penalizeNl;
    }

    /**
     * Returns {@code true} if ignore EOS.
     *
     * @return {@code true} if ignore EOS
     */
    public boolean isIgnoreEos() {
        return ignoreEos;
    }

    /**
     * Returns the mirostat.
     *
     * @return the mirostat
     */
    public int getMirostat() {
        return mirostat;
    }

    /**
     * Returns the mirostat TAU.
     *
     * @return the mirostat TAU
     */
    public float getMirostatTau() {
        return mirostatTau;
    }

    /**
     * Returns the mirostat ETA.
     *
     * @return the mirostat ETA
     */
    public float getMirostatEta() {
        return mirostatEta;
    }

    /**
     * Returns the number of beams.
     *
     * @return the number of beams
     */
    public int getNumberBeams() {
        return nBeams;
    }

    /**
     * Returns the seed.
     *
     * @return the seed
     */
    public int getSeed() {
        return seed;
    }

    /**
     * Returns the logit bias.
     *
     * @return the logit bias
     */
    public Map<Integer, Float> getLogitBias() {
        return logitBias;
    }

    /**
     * Returns the grammar template.
     *
     * @return the grammar template
     */
    public String getGrammar() {
        return grammar;
    }

    /**
     * Returns the anti-prompt.
     *
     * @return the anti-prompt
     */
    public String[] getAntiPrompt() {
        return antiPrompt;
    }
}
