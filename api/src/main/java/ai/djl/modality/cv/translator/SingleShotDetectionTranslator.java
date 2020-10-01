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

import ai.djl.modality.cv.output.BoundingBox;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.output.Rectangle;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.translate.TranslatorContext;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * A {@link BaseImageTranslator} that post-process the {@link NDArray} into {@link DetectedObjects}
 * with boundaries.
 */
public class SingleShotDetectionTranslator extends ObjectDetectionTranslator {

    /**
     * Creates the SSD translator from the given builder.
     *
     * @param builder the builder for the translator
     */
    public SingleShotDetectionTranslator(Builder builder) {
        super(builder);
    }

    /** {@inheritDoc} */
    @Override
    public DetectedObjects processOutput(TranslatorContext ctx, NDList list) {
        float[] classIds = list.get(0).toFloatArray();
        float[] probabilities = list.get(1).toFloatArray();
        NDArray boundingBoxes = list.get(2);

        List<String> retNames = new ArrayList<>();
        List<Double> retProbs = new ArrayList<>();
        List<BoundingBox> retBB = new ArrayList<>();

        for (int i = 0; i < classIds.length; ++i) {
            int classId = (int) classIds[i];
            double probability = probabilities[i];
            // classId starts from 0, -1 means background
            if (classId >= 0 && probability > threshold) {
                if (classId >= classes.size()) {
                    throw new AssertionError("Unexpected index: " + classId);
                }
                String className = classes.get(classId);
                float[] box = boundingBoxes.get(i).toFloatArray();
                // rescale box coordinates by imageWidth and imageHeight
                double x = imageWidth > 0 ? box[0] / imageWidth : box[0];
                double y = imageHeight > 0 ? box[1] / imageHeight : box[1];
                double w = imageWidth > 0 ? box[2] / imageWidth - x : box[2] - x;
                double h = imageHeight > 0 ? box[3] / imageHeight - y : box[3] - y;

                Rectangle rect = new Rectangle(x, y, w, h);
                retNames.add(className);
                retProbs.add(probability);
                retBB.add(rect);
            }
        }

        return new DetectedObjects(retNames, retProbs, retBB);
    }

    /**
     * Creates a builder to build a {@code SingleShotDetectionTranslator}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Creates a builder to build a {@code SingleShotDetectionTranslator} with specified arguments.
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

    /** The builder for SSD translator. */
    public static class Builder extends ObjectDetectionBuilder<Builder> {

        /** {@inheritDoc} */
        @Override
        protected Builder self() {
            return this;
        }

        /**
         * Builds the translator.
         *
         * @return the new translator
         */
        public SingleShotDetectionTranslator build() {
            validate();
            return new SingleShotDetectionTranslator(this);
        }
    }
}
