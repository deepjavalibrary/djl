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
package ai.djl.huggingface.translator;

import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.VisionLanguageInput;
import ai.djl.modality.cv.translator.BaseImagePreProcessor;
import ai.djl.modality.cv.translator.BaseImageTranslator;
import ai.djl.modality.cv.translator.BaseImageTranslator.BaseBuilder;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.translate.ArgumentsUtil;
import ai.djl.translate.NoBatchifyTranslator;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

/** The translator for Huggingface zero-shot-image-classification model. */
public class ZeroShotImageClassificationTranslator
        implements NoBatchifyTranslator<VisionLanguageInput, Classifications> {

    private HuggingFaceTokenizer tokenizer;
    private BaseImageTranslator<?> imageProcessor;
    private boolean int32;

    ZeroShotImageClassificationTranslator(
            HuggingFaceTokenizer tokenizer, BaseImageTranslator<?> imageProcessor, boolean int32) {
        this.tokenizer = tokenizer;
        this.imageProcessor = imageProcessor;
        this.int32 = int32;
    }

    /** {@inheritDoc} */
    @Override
    public NDList processInput(TranslatorContext ctx, VisionLanguageInput input)
            throws TranslateException {
        NDManager manager = ctx.getNDManager();

        String template = input.getHypothesisTemplate();
        String[] candidates = input.getCandidates();
        if (candidates == null || candidates.length == 0) {
            throw new TranslateException("Missing candidates in input");
        }
        List<String> sequences = new ArrayList<>(candidates.length);
        for (String candidate : candidates) {
            sequences.add(applyTemplate(template, candidate));
        }

        Encoding[] encodings = tokenizer.batchEncode(sequences);
        NDList list = Encoding.toNDList(encodings, manager, false, int32);
        Image img = input.getImage();
        NDList imageFeatures = imageProcessor.processInput(ctx, img);
        NDArray array = imageFeatures.get(0).expandDims(0);
        list.add(array);

        ctx.setAttachment("candidates", candidates);
        return list;
    }

    /** {@inheritDoc} */
    @Override
    public Classifications processOutput(TranslatorContext ctx, NDList list)
            throws TranslateException {
        NDArray logits = list.get("logits_per_image");
        logits = logits.squeeze().softmax(0);
        String[] candidates = (String[]) ctx.getAttachment("candidates");
        List<String> classes = Arrays.asList(candidates);
        return new Classifications(classes, logits, candidates.length);
    }

    private String applyTemplate(String template, String arg) {
        int pos = template.indexOf("{}");
        if (pos == -1) {
            return template + arg;
        }
        int len = template.length();
        return template.substring(0, pos) + arg + template.substring(pos + 2, len);
    }

    /**
     * Creates a builder to build a {@code ZeroShotImageClassificationTranslator}.
     *
     * @param tokenizer the tokenizer
     * @return a new builder
     */
    public static Builder builder(HuggingFaceTokenizer tokenizer) {
        return new Builder(tokenizer);
    }

    /**
     * Creates a builder to build a {@code ZeroShotImageClassificationTranslator}.
     *
     * @param tokenizer the tokenizer
     * @param arguments the models' arguments
     * @return a new builder
     */
    public static Builder builder(HuggingFaceTokenizer tokenizer, Map<String, ?> arguments) {
        Builder builder = builder(tokenizer);
        builder.configure(arguments);

        return builder;
    }

    /** The builder for zero-shot classification translator. */
    public static final class Builder extends BaseBuilder<Builder> {

        private HuggingFaceTokenizer tokenizer;
        private boolean int32;

        Builder(HuggingFaceTokenizer tokenizer) {
            this.tokenizer = tokenizer;
        }

        /** {@inheritDoc} */
        @Override
        protected Builder self() {
            return this;
        }

        /**
         * Sets if use int32 datatype for the {@link Translator}.
         *
         * @param int32 true to include token types
         * @return this builder
         */
        public Builder optInt32(boolean int32) {
            this.int32 = int32;
            return this;
        }

        /**
         * Configures the builder with the model arguments.
         *
         * @param arguments the model arguments
         */
        public void configure(Map<String, ?> arguments) {
            configPreProcess(arguments);
            optInt32(ArgumentsUtil.booleanValue(arguments, "int32"));
        }

        /**
         * Builds the translator.
         *
         * @return the new translator
         * @throws IOException if I/O error occurs
         */
        public ZeroShotImageClassificationTranslator build() throws IOException {
            BaseImagePreProcessor processor = new BaseImagePreProcessor(this);
            return new ZeroShotImageClassificationTranslator(tokenizer, processor, int32);
        }
    }
}
