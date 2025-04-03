/*
 * Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.modality.cv.translator;

import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.output.BoundingBox;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.output.Mask;
import ai.djl.modality.cv.transform.ResizeShort;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.translate.ArgumentsUtil;
import ai.djl.translate.TranslatorContext;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * A {@link BaseImageTranslator} that post-process the {@link NDArray} into {@link DetectedObjects}
 * with boundaries at the detailed pixel level.
 */
public class InstanceSegmentationTranslator extends BaseImageTranslator<DetectedObjects> {

    private SynsetLoader synsetLoader;
    private float threshold;

    private List<String> classes;

    /**
     * Creates the Instance Segmentation translator from the given builder.
     *
     * @param builder the builder for the translator
     */
    public InstanceSegmentationTranslator(Builder builder) {
        super(builder);
        this.synsetLoader = builder.synsetLoader;
        this.threshold = builder.threshold;
    }

    /** {@inheritDoc} */
    @Override
    public void prepare(TranslatorContext ctx) throws IOException {
        if (classes == null) {
            classes = synsetLoader.load(ctx.getModel());
        }
    }

    /** {@inheritDoc} */
    @Override
    public DetectedObjects processOutput(TranslatorContext ctx, NDList list) {
        int rescaledWidth = (Integer) ctx.getAttachment("processedWidth");
        int rescaledHeight = (Integer) ctx.getAttachment("processedHeight");

        float[] ids = list.get(0).toFloatArray();
        float[] scores = list.get(1).toFloatArray();
        NDArray boundingBoxes = list.get(2);
        NDArray masks = list.get(3);

        List<String> retNames = new ArrayList<>();
        List<Double> retProbs = new ArrayList<>();
        List<BoundingBox> retBB = new ArrayList<>();

        for (int i = 0; i < ids.length; ++i) {
            int classId = (int) ids[i];
            double probability = scores[i];
            if (classId >= 0 && probability > threshold) {
                if (classId >= classes.size()) {
                    throw new AssertionError("Unexpected index: " + classId);
                }
                String className = classes.get(classId);
                float[] box = boundingBoxes.get(i).toFloatArray();
                double x = box[0] / rescaledWidth;
                double y = box[1] / rescaledHeight;
                double w = box[2] / rescaledWidth - x;
                double h = box[3] / rescaledHeight - y;

                // Reshape mask to actual image bounding box shape.
                NDArray array = masks.get(i);
                float[][] maskFloat = Mask.toMask(array);
                Mask mask = new Mask(x, y, w, h, maskFloat);

                retNames.add(className);
                retProbs.add(probability);
                retBB.add(mask);
            }
        }
        return new DetectedObjects(retNames, retProbs, retBB);
    }

    /**
     * Creates a builder to build a {@code InstanceSegmentationTranslator}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Creates a builder to build a {@code InstanceSegmentationTranslator} with specified arguments.
     *
     * @param arguments arguments to specify builder options
     * @return a new builder
     */
    public static Builder builder(Map<String, ?> arguments) {
        Builder builder = new Builder();
        builder.configPreProcess(arguments);
        builder.configPostProcess(arguments);

        return builder;
    }

    /** The builder for Instance Segmentation translator. */
    public static class Builder extends ClassificationBuilder<Builder> {

        float threshold = 0.3f;
        int shortEdge = 600;
        int maxEdge = 1000;

        Builder() {}

        /**
         * Sets the threshold for prediction accuracy.
         *
         * <p>Predictions below the threshold will be dropped.
         *
         * @param threshold the threshold for prediction accuracy
         * @return the builder
         */
        public Builder optThreshold(float threshold) {
            this.threshold = threshold;
            return this;
        }

        /**
         * Sets the shorter edge length of the rescaled image.
         *
         * @param shortEdge the length of the short edge
         * @return the builder
         */
        public Builder optShortEdge(int shortEdge) {
            this.shortEdge = shortEdge;
            return this;
        }

        /**
         * Sets the maximum edge length of the rescaled image.
         *
         * @param maxEdge the length of the longest edge
         * @return the builder
         */
        public Builder optMaxEdge(int maxEdge) {
            this.maxEdge = maxEdge;
            return this;
        }

        /** {@inheritDoc} */
        @Override
        protected Builder self() {
            return this;
        }

        /** {@inheritDoc} */
        @Override
        protected void configPostProcess(Map<String, ?> arguments) {
            super.configPostProcess(arguments);
            threshold = ArgumentsUtil.floatValue(arguments, "threshold", 0.3f);
            shortEdge = ArgumentsUtil.intValue(arguments, "shortEdge", 600);
            maxEdge = ArgumentsUtil.intValue(arguments, "maxEdge", 1000);
        }

        /**
         * Builds the translator.
         *
         * @return the new translator
         */
        public InstanceSegmentationTranslator build() {
            validate();
            ResizeShort resize = new ResizeShort(shortEdge, maxEdge, Image.Interpolation.BILINEAR);
            pipeline.insert(0, null, resize);

            return new InstanceSegmentationTranslator(this);
        }
    }
}
