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
@SuppressWarnings({"PMD.UnusedPrivateField", "PMD.UnusedAssignment"})
public class InputParameters {

    private int nPredict;
    private int nKeep;
    private int nProbs;
    private int topK;
    private float topP;
    private float tfsZ;
    private float typicalP;
    private float temperature;
    private float repeatPenalty;
    private int repeatLastN;
    private float frequencyPenalty;
    private float presencePenalty;
    private boolean penalizeNl;
    private boolean ignoreEos;
    private int mirostat;
    private float mirostatTau;
    private float mirostatEta;
    private int nBeams;
    private int seed;
    private Map<Integer, Float> logitBias;
    private String grammar;
    private String[] antiPrompt;

    /**
     * Constructs new {@code InputParameters} instance.
     *
     * @param nPredict the max new tokens
     * @param nKeep the number of keep
     * @param nProbs the number of probabilities
     * @param topK the top K
     * @param topP the top P
     * @param tfsZ the tfs Z
     * @param typicalP the typical P
     * @param temperature the temperature
     * @param repeatPenalty the repeat penalty
     * @param repeatLastN the repeat last N
     * @param frequencyPenalty the frequency penalty
     * @param presencePenalty the presence penalty
     * @param penalizeNl the penalize nl
     * @param ignoreEos the ignore EOS
     * @param mirostat the mirostat
     * @param mirostatTau the mirostat TAU
     * @param mirostatEta the mirostat ETA
     * @param nBeams the number of beams
     * @param seed the seed
     * @param logitBias the logit bias
     * @param grammar the grammar
     * @param antiPrompt the anti prompt
     */
    public InputParameters(
            int nPredict,
            int nKeep,
            int nProbs,
            int topK,
            float topP,
            float tfsZ,
            float typicalP,
            float temperature,
            float repeatPenalty,
            int repeatLastN,
            float frequencyPenalty,
            float presencePenalty,
            boolean penalizeNl,
            boolean ignoreEos,
            int mirostat,
            float mirostatTau,
            float mirostatEta,
            int nBeams,
            int seed,
            Map<Integer, Float> logitBias,
            String grammar,
            String[] antiPrompt) {
        this.nPredict = nPredict;
        this.nKeep = nKeep;
        this.nProbs = nProbs;
        this.topK = topK;
        this.topP = topP;
        this.tfsZ = tfsZ;
        this.typicalP = typicalP;
        this.temperature = temperature;
        this.repeatPenalty = repeatPenalty;
        this.repeatLastN = repeatLastN;
        this.frequencyPenalty = frequencyPenalty;
        this.presencePenalty = presencePenalty;
        this.penalizeNl = penalizeNl;
        this.ignoreEos = ignoreEos;
        this.mirostat = mirostat;
        this.mirostatTau = mirostatTau;
        this.mirostatEta = mirostatEta;
        this.nBeams = nBeams;
        this.seed = seed;
        this.logitBias = logitBias;
        this.grammar = grammar;
        this.antiPrompt = antiPrompt;
    }

    /**
     * Returns the max new tokens.
     *
     * @return the max new tokens
     */
    public int getMaxNewTokens() {
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
