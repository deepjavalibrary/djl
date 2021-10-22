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

import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.ndarray.NDArray;
import ai.djl.translate.ArgumentsUtil;
import ai.djl.translate.TranslatorContext;
import java.util.List;
import java.util.Map;

/**
 * A {@link BaseImageTranslator} that post-process the {@link NDArray} into {@link DetectedObjects}
 * with boundaries.
 */
public abstract class ObjectDetectionTranslator extends BaseImageTranslator<DetectedObjects> {

    protected float threshold;
    private SynsetLoader synsetLoader;
    protected List<String> classes;
    protected double imageWidth;
    protected double imageHeight;

    /**
     * Creates the {@link ObjectDetectionTranslator} from the given builder.
     *
     * @param builder the builder for the translator
     */
    protected ObjectDetectionTranslator(ObjectDetectionBuilder<?> builder) {
        super(builder);
        this.threshold = builder.threshold;
        this.synsetLoader = builder.synsetLoader;
        this.imageWidth = builder.imageWidth;
        this.imageHeight = builder.imageHeight;
    }

    /** {@inheritDoc} */
    @Override
    public void prepare(TranslatorContext ctx) throws Exception {
        if (classes == null) {
            classes = synsetLoader.load(ctx.getModel());
        }
    }

    /** The base builder for the object detection translator. */
    @SuppressWarnings("rawtypes")
    public abstract static class ObjectDetectionBuilder<T extends ObjectDetectionBuilder>
            extends ClassificationBuilder<T> {

        protected float threshold = 0.2f;
        protected double imageWidth;
        protected double imageHeight;

        /**
         * Sets the threshold for prediction accuracy.
         *
         * <p>Predictions below the threshold will be dropped.
         *
         * @param threshold the threshold for the prediction accuracy
         * @return this builder
         */
        public T optThreshold(float threshold) {
            this.threshold = threshold;
            return self();
        }

        /**
         * Sets the optional rescale size.
         *
         * @param imageWidth the width to rescale images to
         * @param imageHeight the height to rescale images to
         * @return this builder
         */
        public T optRescaleSize(double imageWidth, double imageHeight) {
            this.imageWidth = imageWidth;
            this.imageHeight = imageHeight;
            return self();
        }

        /**
         * Get resized image width.
         *
         * @return image width
         */
        public double getImageWidth() {
            return imageWidth;
        }

        /**
         * Get resized image height.
         *
         * @return image height
         */
        public double getImageHeight() {
            return imageHeight;
        }

        /** {@inheritDoc} */
        @Override
        protected void configPostProcess(Map<String, ?> arguments) {
            super.configPostProcess(arguments);
            if (ArgumentsUtil.booleanValue(arguments, "rescale")) {
                optRescaleSize(width, height);
            }
            threshold = ArgumentsUtil.floatValue(arguments, "threshold", 0.2f);
        }
    }
}
