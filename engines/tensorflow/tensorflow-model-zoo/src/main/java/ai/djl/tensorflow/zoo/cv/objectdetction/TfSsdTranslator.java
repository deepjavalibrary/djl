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
import ai.djl.modality.cv.translator.ObjectDetectionTranslator;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.translate.ArgumentsUtil;
import ai.djl.translate.TranslatorContext;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

/**
 * A {@link TfSsdTranslator} that post-process the {@link NDArray} into {@link DetectedObjects} with
 * boundaries. Reference implementation: <a
 * href="https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1">SSD</a>.
 */
public class TfSsdTranslator extends ObjectDetectionTranslator {

    private int maxBoxes;
    private String numDetectionsOutputName;
    private String boundingBoxOutputName;
    private String scoresOutputName;
    private String classLabelOutputName;

    /**
     * Creates the SSD translator from the given builder.
     *
     * @param builder the builder for the translator
     */
    protected TfSsdTranslator(Builder builder) {
        super(builder);
        this.maxBoxes = builder.maxBoxes;
        this.numDetectionsOutputName = builder.numDetectionsOutputName;
        this.boundingBoxOutputName = builder.boundingBoxOutputName;
        this.scoresOutputName = builder.scoresOutputName;
        this.classLabelOutputName = builder.classLabelOutputName;
    }

    /** {@inheritDoc} */
    @Override
    public NDList processInput(TranslatorContext ctx, Image input) {
        // TensorFlow object detection model does not support batch input
        // and require input dimension to be 4 with the first dim equals to 1,
        // remove batchifier and manually batch input in preprocess. See:
        // https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1
        return new NDList(super.processInput(ctx, input).get(0).expandDims(0));
    }

    /** {@inheritDoc} */
    @Override
    public DetectedObjects processOutput(TranslatorContext ctx, NDList list) {
        int len = (int) list.get(0).getShape().get(0);
        for (NDArray array : list) {
            if (numDetectionsOutputName.equals(array.getName())) {
                len = array.toArray()[0].intValue();
                break;
            }
        }
        float[] scores = new float[len];
        long[] classIds = new long[len];

        NDArray boundingBoxes = list.get(0);
        for (NDArray array : list) {
            if (scoresOutputName.equals(array.getName())) {
                scores = array.toFloatArray();
            } else if (boundingBoxOutputName.equals(array.getName())) {
                boundingBoxes = array;
            } else if (classLabelOutputName.equals(array.getName())) {
                classIds = Arrays.stream(array.toArray()).mapToLong(Number::longValue).toArray();
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
            if (classId >= 0 && score > threshold) {
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
     * Creates a builder to build a {@code TfSSDTranslator}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Creates a builder to build a {@code TfSSDTranslator} with specified arguments.
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

    /** The builder for TensorFlow SSD translator. */
    public static class Builder extends ObjectDetectionBuilder<Builder> {

        int maxBoxes = 10;
        String numDetectionsOutputName = "num_detections";
        String boundingBoxOutputName = "detection_boxes";
        String scoresOutputName = "detection_scores";
        String classLabelOutputName = "detection_class_labels";

        /**
         * Set the output name used for number of detections.
         *
         * <p>You can find the output names of TensorFlow models by calling `model.describeOutput()`
         * after loading it.
         *
         * @param numDetectionsOutputName output name for number of detections
         * @return this builder
         */
        public Builder optNumDetectionsOutputName(String numDetectionsOutputName) {
            this.numDetectionsOutputName = numDetectionsOutputName;
            return this;
        }

        /**
         * Set the output name used for bounding boxes. You can find the output names of TensorFlow
         * models by calling `model.describeOutput()` after loading it.
         *
         * @param boundingBoxOutputName output name for bounding boxes
         * @return this builder
         */
        public Builder optBoundingBoxOutputName(String boundingBoxOutputName) {
            this.boundingBoxOutputName = boundingBoxOutputName;
            return this;
        }

        /**
         * Set the output name used for detection scores. You can find the output names of
         * TensorFlow models by calling `model.describeOutput()` after loading it.
         *
         * @param scoresOutputName output name for detection scores
         * @return this builder
         */
        public Builder optScoresOutputName(String scoresOutputName) {
            this.scoresOutputName = scoresOutputName;
            return this;
        }

        /**
         * Set the output name used for class label. You can find the output names of TensorFlow
         * models by calling `model.describeOutput()` after loading it.
         *
         * @param classLabelOutputName output name for class label
         * @return this builder
         */
        public Builder optClassLabelOutputName(String classLabelOutputName) {
            this.classLabelOutputName = classLabelOutputName;
            return this;
        }

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

        /** {@inheritDoc} */
        @Override
        protected Builder self() {
            return this;
        }

        /** {@inheritDoc} */
        @Override
        protected void configPreProcess(Map<String, ?> arguments) {
            super.configPreProcess(arguments);
            optBatchifier(null); // override parent batchifier
        }

        /** {@inheritDoc} */
        @Override
        protected void configPostProcess(Map<String, ?> arguments) {
            super.configPostProcess(arguments);
            maxBoxes = ArgumentsUtil.intValue(arguments, "maxBoxes", 10);
            threshold = ArgumentsUtil.floatValue(arguments, "threshold", 0.4f);
            numDetectionsOutputName =
                    ArgumentsUtil.stringValue(
                            arguments, "numDetectionsOutputName", "num_detections");
            boundingBoxOutputName =
                    ArgumentsUtil.stringValue(
                            arguments, "boundingBoxOutputName", "detection_boxes");
            scoresOutputName =
                    ArgumentsUtil.stringValue(arguments, "scoresOutputName", "detection_scores");
            classLabelOutputName =
                    ArgumentsUtil.stringValue(
                            arguments, "classLabelOutputName", "detection_class_labels");
        }

        /**
         * Builds the translator.
         *
         * @return the new translator
         */
        public TfSsdTranslator build() {
            validate();
            return new TfSsdTranslator(this);
        }
    }
}
