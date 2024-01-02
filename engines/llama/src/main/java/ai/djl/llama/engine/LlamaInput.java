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
package ai.djl.llama.engine;

import ai.djl.llama.jni.InputParameters;

import com.google.gson.annotations.SerializedName;

import java.util.Map;

/** A class hold input data for Llama model. */
public class LlamaInput {

    private String inputs;
    private String prefix;
    private String suffix;
    private Parameters parameters;

    /**
     * Returns the input prompt.
     *
     * @return the input prompt
     */
    public String getInputs() {
        return inputs;
    }

    /**
     * Sets the input prompt.
     *
     * @param inputs the input prompt
     */
    public void setInputs(String inputs) {
        this.inputs = inputs;
    }

    /**
     * Returns the prompt prefix.
     *
     * @return the prompt prefix
     */
    public String getPrefix() {
        return prefix;
    }

    /**
     * Sets the prompt prefix.
     *
     * @param prefix the prompt prefix
     */
    public void setPrefix(String prefix) {
        this.prefix = prefix;
    }

    /**
     * Returns the prompt suffix.
     *
     * @return the prompt suffix
     */
    public String getSuffix() {
        return suffix;
    }

    /**
     * Sets the prompt suffix.
     *
     * @param suffix the prompt suffix
     */
    public void setSuffix(String suffix) {
        this.suffix = suffix;
    }

    /**
     * Returns the input parameters.
     *
     * @return the input parameters
     */
    public Parameters getParameters() {
        if (parameters == null) {
            parameters = new Parameters();
        }
        return parameters;
    }

    /**
     * Sets the input parameters.
     *
     * @param parameters the input parameters
     */
    public void setParameters(Parameters parameters) {
        this.parameters = parameters;
    }

    /** The input parameters class. */
    public static final class Parameters {

        @SerializedName("max_new_tokens")
        private int nPredict;

        @SerializedName("number_keep")
        private int nKeep;

        @SerializedName("number_probabilities")
        private int nProbs;

        @SerializedName("top_k")
        private int topK;

        @SerializedName("top_p")
        private float topP;

        @SerializedName("tfs_z")
        private float tfsZ;

        @SerializedName("typical_p")
        private float typicalP;

        @SerializedName("temperature")
        private float temperature;

        @SerializedName("repeat_penalty")
        private float repeatPenalty;

        @SerializedName("repeat_last_n")
        private int repeatLastN;

        @SerializedName("frequency_penalty")
        private float frequencyPenalty;

        @SerializedName("presence_penalty")
        private float presencePenalty;

        @SerializedName("penalize_nl")
        private boolean penalizeNl;

        @SerializedName("ignore_eos")
        private boolean ignoreEos;

        @SerializedName("mirostat")
        private int mirostat;

        @SerializedName("mirostat_tau")
        private float mirostatTau;

        @SerializedName("mirostat_eta")
        private float mirostatEta;

        @SerializedName("number_beams")
        private int nBeams;

        @SerializedName("seed")
        private int seed;

        @SerializedName("logit_bias")
        private Map<Integer, Float> logitBias;

        @SerializedName("grammar")
        private String grammar;

        @SerializedName("anti_prompt")
        private String[] antiPrompt;

        /**
         * Sets the max new tokens.
         *
         * @param maxNewTokens the max new tokens
         */
        public void setMaxNewTokens(int maxNewTokens) {
            this.nPredict = maxNewTokens;
        }

        /**
         * Sets the number of keep.
         *
         * @param nKeep the number of keep
         */
        public void setNumberKeep(int nKeep) {
            this.nKeep = nKeep;
        }

        /**
         * Sets the number of probabilities.
         *
         * @param nProbs the number of probabilities
         */
        public void setNumberProbabilities(int nProbs) {
            this.nProbs = nProbs;
        }

        /**
         * Sets the top K.
         *
         * @param topK the top K
         */
        public void setTopK(int topK) {
            this.topK = topK;
        }

        /**
         * Sets the top P.
         *
         * @param topP the top P
         */
        public void setTopP(float topP) {
            this.topP = topP;
        }

        /**
         * Sets the tfs Z.
         *
         * @param tfsZ the tfs Z
         */
        public void setTfsZ(float tfsZ) {
            this.tfsZ = tfsZ;
        }

        /**
         * Sets the typical P.
         *
         * @param typicalP the typical P
         */
        public void setTypicalP(float typicalP) {
            this.typicalP = typicalP;
        }

        /**
         * Sets the temperature.
         *
         * @param temperature the temperature
         */
        public void setTemperature(float temperature) {
            this.temperature = temperature;
        }

        /**
         * Sets the repeat penalty.
         *
         * @param repeatPenalty the repeat penalty
         */
        public void setRepeatPenalty(float repeatPenalty) {
            this.repeatPenalty = repeatPenalty;
        }

        /**
         * Sets the repeat last N.
         *
         * @param repeatLastN the repeat last N
         */
        public void setRepeatLastN(int repeatLastN) {
            this.repeatLastN = repeatLastN;
        }

        /**
         * Sets the frequency penalty.
         *
         * @param frequencyPenalty the frequency penalty
         */
        public void setFrequencyPenalty(float frequencyPenalty) {
            this.frequencyPenalty = frequencyPenalty;
        }

        /**
         * Sets the presence penalty.
         *
         * @param presencePenalty the presence penalty
         */
        public void setPresencePenalty(float presencePenalty) {
            this.presencePenalty = presencePenalty;
        }

        /**
         * Sets the penalize nl.
         *
         * @param penalizeNl the penalize nl
         */
        public void setPenalizeNl(boolean penalizeNl) {
            this.penalizeNl = penalizeNl;
        }

        /**
         * Sets if ignore EOS.
         *
         * @param ignoreEos if ignore EOS
         */
        public void setIgnoreEos(boolean ignoreEos) {
            this.ignoreEos = ignoreEos;
        }

        /**
         * Sets the mirostat.
         *
         * @param mirostat the mirostat
         */
        public void setMirostat(int mirostat) {
            this.mirostat = mirostat;
        }

        /**
         * Sets the mirostat TAU.
         *
         * @param mirostatTau the mirostat TAU
         */
        public void setMirostatTau(float mirostatTau) {
            this.mirostatTau = mirostatTau;
        }

        /**
         * Sets the mirostat ETA.
         *
         * @param mirostatEta the mirostat ETA
         */
        public void setMirostatEta(float mirostatEta) {
            this.mirostatEta = mirostatEta;
        }

        /**
         * Sets the number of beams.
         *
         * @param nBeams the number of beams
         */
        public void setNumberBeams(int nBeams) {
            this.nBeams = nBeams;
        }

        /**
         * Sets the seed.
         *
         * @param seed the seed
         */
        public void setSeed(int seed) {
            this.seed = seed;
        }

        /**
         * Sets the logit bias.
         *
         * @param logitBias the logit bias
         */
        public void setLogitBias(Map<Integer, Float> logitBias) {
            this.logitBias = logitBias;
        }

        /**
         * Sets the grammar template.
         *
         * @param grammar the grammar template
         */
        public void setGrammar(String grammar) {
            this.grammar = grammar;
        }

        /**
         * Sets the anti prompt.
         *
         * @param antiPrompt the anti prompt
         */
        public void setAntiPrompt(String[] antiPrompt) {
            this.antiPrompt = antiPrompt;
        }

        /**
         * Returns the {@link InputParameters} object.
         *
         * @return the {@link InputParameters} object
         */
        public InputParameters toInputParameters() {
            setDefaultValue();
            return new InputParameters(
                    nPredict,
                    nKeep,
                    nProbs,
                    topK,
                    topP,
                    tfsZ,
                    typicalP,
                    temperature,
                    repeatPenalty,
                    repeatLastN,
                    frequencyPenalty,
                    presencePenalty,
                    penalizeNl,
                    ignoreEos,
                    mirostat,
                    mirostatTau,
                    mirostatEta,
                    nBeams,
                    seed,
                    logitBias,
                    grammar,
                    antiPrompt);
        }

        private void setDefaultValue() {
            if (nPredict == 0) {
                nPredict = -1;
            }
            if (topK == 0) {
                topK = 40;
            }
            if (topP == 0) {
                topP = 0.95f;
            }
            if (tfsZ == 0) {
                tfsZ = 1f;
            }
            if (typicalP == 0) {
                typicalP = 1f;
            }
            if (temperature == 0) {
                temperature = 0.8f;
            }
            if (repeatPenalty == 0) {
                repeatPenalty = 1.10f;
            }
            if (repeatLastN == 0) {
                repeatLastN = 64;
            }
        }
    }
}
