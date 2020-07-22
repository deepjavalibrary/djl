/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.tensorflow.zoo.cv.objectdetction;

import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.output.BoundingBox;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.output.Rectangle;
import ai.djl.modality.cv.translator.SingleShotDetectionTranslator;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.DataType;
import ai.djl.translate.Batchifier;
import ai.djl.translate.TranslatorContext;
import java.util.ArrayList;
import java.util.List;

/**
 * A {@link TfSsdTranslator} that post-process the {@link NDArray} into {@link DetectedObjects} with
 * boundaries. Reference implementation: <a
 * href="https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Detection/SSD">SSD</a>.
 */
public class TfSsdTranslator extends SingleShotDetectionTranslator {

    private int maxBoxes;
    private float threshHold;

    /**
     * Creates the SSD translator from the given builder.
     *
     * @param builder the builder for the translator
     */
    protected TfSsdTranslator(Builder builder) {
        super(builder);
        this.maxBoxes = builder.maxBoxes;
        this.threshHold = builder.getThreshold();
    }

    @Override
    public NDList processInput(TranslatorContext ctx, Image input) {
        // TensorFlow object detection model does not support batch input
        // and require input dimension to be 4 with the first dim equals to 1,
        // remove batchifier and manually batch input in preprocess. See:
        // https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1
        return new NDList(super.processInput(ctx, input).get(0).expandDims(0));
    }

    @Override
    public Batchifier getBatchifier() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public DetectedObjects processOutput(TranslatorContext ctx, NDList list) {
        // output orders are not guaranteed
        int len = (int) list.get(0).getShape().get(0);
        float[] scores = new float[len];
        long[] classIds = new long[len];

        NDArray boundingBoxes = list.get(0);
        for (NDArray array : list) {
            DataType dType = array.getDataType();
            int dim = array.getShape().dimension();
            if (dType == DataType.FLOAT32 && dim == 1) {
                scores = array.toFloatArray();
            } else if (dType == DataType.FLOAT32 && dim == 2) {
                boundingBoxes = array;
            } else if (dType == DataType.INT64 && dim == 1) {
                classIds = array.toLongArray();
            } else {
                throw new IllegalStateException(
                        "Unexpected result NDArray type:" + dType + ", and dim: " + dim);
            }
        }
        List<String> retNames = new ArrayList<>();
        List<Double> retProbs = new ArrayList<>();
        List<BoundingBox> retBB = new ArrayList<>();

        // results are already sorted according to scores
        for (int i = 0; i < Math.min(classIds.length, maxBoxes); ++i) {
            long classId = classIds[i];
            double score = scores[i];
            // classId starts from 0, -1 means background
            if (classId >= 0 && score > threshHold) {
                if (classId >= classes.size()) {
                    throw new AssertionError("Unexpected index: " + classId);
                }
                String className = classes.get((int) classId - 1);
                float[] box = boundingBoxes.get(i).toFloatArray();
                float yMin = box[0];
                float xMin = box[1];
                float yMax = box[2];
                float xMax = box[3];
                double w = xMax - xMin;
                double h = yMax - yMin;
                Rectangle rect = new Rectangle(xMin, yMin, w, h);
                retNames.add(className);
                retProbs.add(score);
                retBB.add(rect);
            }
        }

        return new DetectedObjects(retNames, retProbs, retBB);
    }

    /**
     * Creates a builder to build a {@code TfSSDTranslatorBuilder}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /** The builder for TensorFlow SSD translator. */
    public static class Builder extends SingleShotDetectionTranslator.Builder {

        private int maxBoxes = 10;

        /**
         * Set the maximum number of bounding boxes to display.
         *
         * @param maxBoxes maximum number of bounding boxes to display
         * @return this builder
         */
        public Builder optMaxBoxes(int maxBoxes) {
            this.maxBoxes = maxBoxes;
            return this;
        }

        /**
         * Builds the translator.
         *
         * @return the new translator
         */
        @Override
        public TfSsdTranslator build() {
            validate();
            return new TfSsdTranslator(this);
        }
    }
}
