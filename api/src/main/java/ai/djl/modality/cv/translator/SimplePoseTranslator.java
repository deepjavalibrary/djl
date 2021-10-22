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

import ai.djl.modality.cv.output.Joints;
import ai.djl.modality.cv.output.Joints.Joint;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.translate.ArgumentsUtil;
import ai.djl.translate.TranslatorContext;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * A {@link BaseImageTranslator} that post-process the {@link NDArray} into human {@link Joints}.
 */
public class SimplePoseTranslator extends BaseImageTranslator<Joints> {

    private float threshold;

    /**
     * Creates the Pose Estimation translator from the given builder.
     *
     * @param builder the builder for the translator
     */
    public SimplePoseTranslator(Builder builder) {
        super(builder);
        this.threshold = builder.threshold;
    }

    /** {@inheritDoc} */
    @Override
    public Joints processOutput(TranslatorContext ctx, NDList list) {
        NDArray pred = list.singletonOrThrow();
        int numJoints = (int) pred.getShape().get(0);
        int height = (int) pred.getShape().get(1);
        int width = (int) pred.getShape().get(2);
        NDArray predReshaped = pred.reshape(new Shape(1, numJoints, -1));
        NDArray maxIndices =
                predReshaped
                        .argMax(2)
                        .reshape(new Shape(1, numJoints, -1))
                        .toType(DataType.FLOAT32, false);
        NDArray maxValues = predReshaped.max(new int[] {2}, true);

        NDArray result = maxIndices.tile(2, 2);

        result.set(new NDIndex(":, :, 0"), result.get(":, :, 0").mod(width));
        result.set(new NDIndex(":, :, 1"), result.get(":, :, 1").div(width).floor());
        // TODO remove asType
        NDArray predMask =
                maxValues
                        .gt(0.0)
                        // current boolean NDArray operator didn't support majority of ops
                        // need to cast to int
                        .toType(DataType.UINT8, false)
                        .tile(2, 2)
                        .toType(DataType.BOOLEAN, false);
        float[] flattened = result.get(predMask).toFloatArray();
        float[] flattenedConfidence = maxValues.toFloatArray();
        List<Joint> joints = new ArrayList<>(numJoints);
        for (int i = 0; i < numJoints; ++i) {
            if (flattenedConfidence[i] > threshold) {
                joints.add(
                        new Joint(
                                flattened[i * 2] / width,
                                flattened[i * 2 + 1] / height,
                                flattenedConfidence[i]));
            }
        }
        return new Joints(joints);
    }

    /**
     * Creates a builder to build a {@code SimplePoseTranslator}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Creates a builder to build a {@code SimplePoseTranslator} with specified arguments.
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

    /** The builder for Pose Estimation translator. */
    public static class Builder extends BaseBuilder<Builder> {

        float threshold = 0.2f;

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
            return self();
        }

        /** {@inheritDoc} */
        @Override
        protected Builder self() {
            return this;
        }

        /** {@inheritDoc} */
        @Override
        protected void configPostProcess(Map<String, ?> arguments) {
            threshold = ArgumentsUtil.floatValue(arguments, "threshold", 0.2f);
        }

        /**
         * Builds the translator.
         *
         * @return the new translator
         */
        public SimplePoseTranslator build() {
            validate();
            return new SimplePoseTranslator(this);
        }
    }
}
