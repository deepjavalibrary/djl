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
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.VisionLanguageInput;
import ai.djl.modality.cv.output.BoundingBox;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.output.Rectangle;
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
import java.util.List;
import java.util.Map;

/** The translator for Huggingface zero-shot-object-detection model. */
public class ZeroShotObjectDetectionTranslator
        implements NoBatchifyTranslator<VisionLanguageInput, DetectedObjects> {

    private HuggingFaceTokenizer tokenizer;
    private BaseImageTranslator<?> imageProcessor;
    private boolean int32;
    private float threshold;

    ZeroShotObjectDetectionTranslator(
            HuggingFaceTokenizer tokenizer,
            BaseImageTranslator<?> imageProcessor,
            boolean int32,
            float threshold) {
        this.tokenizer = tokenizer;
        this.imageProcessor = imageProcessor;
        this.int32 = int32;
        this.threshold = threshold;
    }

    /** {@inheritDoc} */
    @Override
    public NDList processInput(TranslatorContext ctx, VisionLanguageInput input)
            throws TranslateException {
        NDManager manager = ctx.getNDManager();

        String[] candidates = input.getCandidates();
        if (candidates == null || candidates.length == 0) {
            throw new TranslateException("Missing candidates in input");
        }

        Encoding[] encodings = tokenizer.batchEncode(candidates);
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
    public DetectedObjects processOutput(TranslatorContext ctx, NDList list)
            throws TranslateException {
        NDArray logits = list.get("logits");
        NDArray boxes = list.get("pred_boxes");
        NDArray labels = logits.argMax(-1);
        NDArray scores = logits.max(new int[] {-1}).getNDArrayInternal().sigmoid();
        NDArray selected = scores.gt(threshold);

        scores = scores.get(selected);
        labels = labels.get(selected);
        boxes = boxes.get(selected);

        float[] prob = scores.toFloatArray();
        long[] labelsIndex = labels.toLongArray();
        float[] box = boxes.toFloatArray();

        String[] candidates = (String[]) ctx.getAttachment("candidates");

        List<String> classes = new ArrayList<>(labelsIndex.length);
        List<Double> probabilities = new ArrayList<>(labelsIndex.length);
        List<BoundingBox> boundingBoxes = new ArrayList<>(labelsIndex.length);
        int width = (Integer) ctx.getAttachment("width");
        int height = (Integer) ctx.getAttachment("height");

        for (int i = 0; i < labelsIndex.length; i++) {
            classes.add(candidates[(int) labelsIndex[i]]);
            int pos = i * 4;
            float x = box[pos];
            float y = box[pos + 1];
            float w = box[pos + 2];
            float h = box[pos + 3];
            x = x - w / 2;
            y = y - h / 2;
            // remove padding stretch
            if (width > height) {
                y = y * width / height;
                h = h * width / height;
            } else if (width < height) {
                x = x * height / width;
                w = w * height / width;
            }

            BoundingBox bbox = new Rectangle(x, y, w, h);
            boundingBoxes.add(bbox);
            probabilities.add((double) prob[i]);
        }
        return new DetectedObjects(classes, probabilities, boundingBoxes);
    }

    /**
     * Creates a builder to build a {@code ZeroShotObjectDetectionTranslator}.
     *
     * @param tokenizer the tokenizer
     * @return a new builder
     */
    public static Builder builder(HuggingFaceTokenizer tokenizer) {
        return new Builder(tokenizer);
    }

    /**
     * Creates a builder to build a {@code ZeroShotObjectDetectionTranslator}.
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
        private float threshold = 0.2f;

        Builder(HuggingFaceTokenizer tokenizer) {
            this.tokenizer = tokenizer;
        }

        /** {@inheritDoc} */
        @Override
        protected Builder self() {
            return this;
        }

        /**
         * Sets the threshold for prediction accuracy.
         *
         * <p>Predictions below the threshold will be dropped.
         *
         * @param threshold the threshold for the prediction accuracy
         * @return this builder
         */
        public Builder optThreshold(float threshold) {
            this.threshold = threshold;
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
            optThreshold(ArgumentsUtil.floatValue(arguments, "threshold", 0.2f));
        }

        /**
         * Builds the translator.
         *
         * @return the new translator
         * @throws IOException if I/O error occurs
         */
        public ZeroShotObjectDetectionTranslator build() throws IOException {
            BaseImageTranslator<?> imageProcessor = new BaseImagePreProcessor(this);
            return new ZeroShotObjectDetectionTranslator(
                    tokenizer, imageProcessor, int32, threshold);
        }
    }
}
