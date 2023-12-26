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

import ai.djl.inference.streaming.IteratorBytesSupplier;
import ai.djl.llama.jni.InputParameters;
import ai.djl.llama.jni.LlamaLibrary;
import ai.djl.llama.jni.Token;
import ai.djl.llama.jni.TokenIterator;
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.ndarray.BytesSupplier;
import ai.djl.ndarray.NDList;
import ai.djl.translate.NoBatchifyTranslator;
import ai.djl.translate.TranslatorContext;
import ai.djl.util.JsonUtils;

import com.google.gson.annotations.SerializedName;

import java.util.Iterator;
import java.util.Map;

/** Built-in {@code Translator} that provides preprocessing and postprocessing for llama.cpp. */
public class LlamaTranslator<I, O> implements NoBatchifyTranslator<I, O> {

    private long handle;

    /** {@inheritDoc} */
    @Override
    public void prepare(TranslatorContext ctx) {
        LlamaModel model = (LlamaModel) ctx.getModel();
        handle = model.getHandle();
    }

    /** {@inheritDoc} */
    @Override
    public NDList processInput(TranslatorContext ctx, I input) {
        if (input instanceof String) {
            ctx.setAttachment("out", generate((String) input));
        } else if (input instanceof Input) {
            String prompt = ((Input) input).getData().getAsString();
            TokenIterator it = generate(prompt);
            Output output = new Output();
            output.add(new IteratorBytesSupplier(new OutputIterator(it)));
            ctx.setAttachment("out", output);
        }
        return new NDList();
    }

    /** {@inheritDoc} */
    @Override
    @SuppressWarnings("unchecked")
    public O processOutput(TranslatorContext ctx, NDList list) {
        return (O) ctx.getAttachment("out");
    }

    private TokenIterator generate(String input) {
        TextGenerationInput in = JsonUtils.GSON.fromJson(input, TextGenerationInput.class);
        InputParameters param = in.parameters.toInputParameters();
        if (in.prefix != null && in.suffix != null) {
            LlamaLibrary.infill(handle, in.prefix, in.prefix, param);
        } else if (in.inputs != null && !in.inputs.isEmpty()) {
            LlamaLibrary.generate(handle, in.inputs, param);
        } else {
            throw new IllegalArgumentException("Unsupported input format");
        }
        return new TokenIterator(handle);
    }

    private static final class OutputIterator implements Iterator<BytesSupplier> {

        private TokenIterator it;

        public OutputIterator(TokenIterator it) {
            this.it = it;
        }

        /** {@inheritDoc} */
        @Override
        public boolean hasNext() {
            return it.hasNext();
        }

        /** {@inheritDoc} */
        @Override
        public BytesSupplier next() {
            Token token = it.next();
            return BytesSupplier.wrap(JsonUtils.GSON.toJson(token) + "\n");
        }
    }

    protected static final class TextGenerationInput {

        String inputs;
        String prefix;
        String suffix;
        Parameters parameters;
    }

    protected static final class Parameters {

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
