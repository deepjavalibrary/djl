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
import ai.djl.ndarray.types.DataType;
import ai.djl.translate.TranslatorContext;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/** A translator for yolo models. */
public class YoloTranslator extends ObjectDetectionTranslator {

    /**
     * Constructs an ImageTranslator with the provided builder.
     *
     * @param builder the data to build with
     */
    public YoloTranslator(Builder builder) {
        super(builder);
    }

    /** {@inheritDoc} */
    @Override
    public DetectedObjects processOutput(TranslatorContext ctx, NDList list) {
        int[] classIndices = list.get(0).toType(DataType.INT32, true).flatten().toIntArray();
        double[] probs = list.get(1).toType(DataType.FLOAT64, true).flatten().toDoubleArray();
        NDArray boundingBoxes = list.get(2);
        int detected = Math.toIntExact(probs.length);

        NDArray xMin = boundingBoxes.get(":, 0").clip(0, imageWidth).div(imageWidth);
        NDArray yMin = boundingBoxes.get(":, 1").clip(0, imageHeight).div(imageHeight);
        NDArray xMax = boundingBoxes.get(":, 2").clip(0, imageWidth).div(imageWidth);
        NDArray yMax = boundingBoxes.get(":, 3").clip(0, imageHeight).div(imageHeight);

        float[] boxX = xMin.toFloatArray();
        float[] boxY = yMin.toFloatArray();
        float[] boxWidth = xMax.sub(xMin).toFloatArray();
        float[] boxHeight = yMax.sub(yMin).toFloatArray();

        List<String> retClasses = new ArrayList<>(detected);
        List<Double> retProbs = new ArrayList<>(detected);
        List<BoundingBox> retBB = new ArrayList<>(detected);
        for (int i = 0; i < detected; i++) {
            if (classIndices[i] < 0 || probs[i] < threshold) {
                continue;
            }
            retClasses.add(classes.get(classIndices[i]));
            retProbs.add(probs[i]);
            Rectangle rect = new Rectangle(boxX[i], boxY[i], boxWidth[i], boxHeight[i]);
            retBB.add(rect);
        }
        return new DetectedObjects(retClasses, retProbs, retBB);
    }

    /**
     * Creates a builder to build a {@link YoloTranslator}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Creates a builder to build a {@code YoloTranslator} with specified arguments.
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

    /** The builder for {@link YoloTranslator}. */
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
        public YoloTranslator build() {
            validate();
            return new YoloTranslator(this);
        }
    }
}
