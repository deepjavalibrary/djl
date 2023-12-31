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
package ai.djl.modality.cv.translator;

import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.output.Rectangle;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.translate.ArgumentsUtil;

import java.util.ArrayList;
import java.util.Map;

/**
 * A translator for YoloV8 models. This was tested with ONNX exported Yolo models. For details check
 * here: https://github.com/ultralytics/ultralytics
 */
public class YoloV8Translator extends YoloV5Translator {

    int maxBoxes;

    /**
     * Constructs an ImageTranslator with the provided builder.
     *
     * @param builder the data to build with
     */
    protected YoloV8Translator(Builder builder) {
        super(builder);
        maxBoxes = builder.maxBox;
    }

    /**
     * Creates a builder to build a {@code YoloV8Translator} with specified arguments.
     *
     * @param arguments arguments to specify builder options
     * @return a new builder
     */
    public static YoloV8Translator.Builder builder(Map<String, ?> arguments) {
        YoloV8Translator.Builder builder = new YoloV8Translator.Builder();
        builder.configPreProcess(arguments);
        builder.configPostProcess(arguments);

        return builder;
    }

    @Override
    protected DetectedObjects processFromBoxOutput(NDList list) {
        NDArray rawResult = list.get(0);
        NDArray reshapedResult = rawResult.transpose();
        Shape preparedResult = reshapedResult.getShape();
        long numberRows = preparedResult.get(0);
        long sizeClasses = preparedResult.get(1);

        ArrayList<IntermediateResult> intermediateResults = new ArrayList<>();
        // reverse order search in heap; searches through #maxBoxes for optimization when set
        for (int i = (int) numberRows - 1; i > numberRows - maxBoxes; i--) {
            final float[] row = reshapedResult.get(i).toFloatArray();

            float maxClassProb = -1f;
            int maxIndex = -1;
            for (int c = 4; c < sizeClasses; c++) {
                float classProb = row[c];
                if (classProb > maxClassProb) {
                    maxClassProb = classProb;
                    maxIndex = c;
                }
            }

            if (maxClassProb > threshold) {
                float xPos = row[0]; // center x
                float yPos = row[1]; // center y
                float w = row[2];
                float h = row[3];
                final Rectangle rect =
                        new Rectangle(Math.max(0, xPos - w / 2), Math.max(0, yPos - h / 2), w, h);
                intermediateResults.add(
                        new IntermediateResult(
                                classes.get(maxIndex), maxClassProb, maxIndex, rect));
            }
        }
        return nms(intermediateResults);
    }

    /** The builder for {@link YoloV8Translator}. */
    public static class Builder extends YoloV5Translator.Builder {
        int maxBox = 8400;
        /**
         * Builds the translator.
         *
         * @return the new translator
         */
        @Override
        public YoloV8Translator build() {
            if (pipeline == null) {
                addTransform(
                        array -> array.transpose(2, 0, 1).toType(DataType.FLOAT32, false).div(255));
            }
            validate();
            return new YoloV8Translator(this);
        }

        @Override
        protected void configPostProcess(Map<String, ?> arguments) {
            super.configPostProcess(arguments);
            maxBox = ArgumentsUtil.intValue(arguments, "maxBox", 8400);
        }
    }
}
