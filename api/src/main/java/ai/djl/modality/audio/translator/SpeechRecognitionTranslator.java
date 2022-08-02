/*
 * Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

package ai.djl.modality.audio.translator;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.translate.ArgumentsUtil;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;

import java.nio.FloatBuffer;
import java.util.Map;

/**
 * A {@link Translator} that post-process the {@link NDList} into {@link NDList} to get a text
 * translation of the audio.
 */
public class SpeechRecognitionTranslator implements Translator<NDList, NDList> {

    private final int audioLenInSecond;
    private final int sampleRate;
    private final int recordingLength;

    /**
     * Creates the Speech Recognition translator from the given builder.
     *
     * @param builder the builder for the translator
     */
    public SpeechRecognitionTranslator(Builder builder) {
        this.audioLenInSecond = builder.audioLenInSecond;
        this.sampleRate = builder.sampleRate;
        this.recordingLength = builder.recordingLength;
    }

    /** {@inheritDoc} */
    @Override
    public NDList processInput(TranslatorContext ctx, NDList input) throws Exception {
        try (NDManager manager = NDManager.newBaseManager()) {
            float[] floatInputBuffer = input.get(0).toFloatArray();
            double[] wav2vecinput = new double[recordingLength];
            for (int n = 0; n < recordingLength; n++) {
                wav2vecinput[n] = floatInputBuffer[n] / (float) Short.MAX_VALUE;
            }

            FloatBuffer inTensorBuffer = FloatBuffer.allocate(recordingLength);
            for (double val : wav2vecinput) {
                inTensorBuffer.put((float) val);
            }

            NDArray inTensor = manager.create(inTensorBuffer, new Shape(1, recordingLength));
            return new NDList(inTensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDList processOutput(TranslatorContext ctx, NDList list) throws Exception {
        return list;
    }

    /**
     * Creates a builder to build a {@code SpeechRecognitionTranslator}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Creates a builder to build a {@code SpeechRecognitionTranslator} with specified arguments.
     *
     * @param arguments arguments to specify builder options
     * @return a new builder
     */
    public static Builder builder(Map<String, ?> arguments) {
        Builder builder = new Builder();

        builder.configPreProcess(arguments);

        return builder;
    }

    /** The builder for Speech Recognition translator. */
    public static class Builder {
        int audioLenInSecond = 6;
        int sampleRate = 16000;
        int recordingLength = audioLenInSecond * sampleRate;

        Builder() {}

        /**
         * Builder for Speech Recognition.
         *
         * @return Speech Recognition Builder
         */
        protected Builder self() {
            return this;
        }

        /**
         * Process inputs for Speech Recognition Translator.
         *
         * @param arguments contains the specifications for this builder
         */
        protected void configPreProcess(Map<String, ?> arguments) {
            audioLenInSecond = ArgumentsUtil.intValue(arguments, "audioLenInSecond", 6);
            sampleRate = ArgumentsUtil.intValue(arguments, "sampleRate", 16000);
            recordingLength = sampleRate * audioLenInSecond;
        }

        /**
         * Builds the translator.
         *
         * @return the new translator
         */
        public SpeechRecognitionTranslator build() {
            return new SpeechRecognitionTranslator(this);
        }
    }
}
