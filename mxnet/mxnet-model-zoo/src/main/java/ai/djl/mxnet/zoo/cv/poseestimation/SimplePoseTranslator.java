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
package ai.djl.mxnet.zoo.cv.poseestimation;

import ai.djl.modality.cv.ImageTranslator;
import ai.djl.modality.cv.Joints;
import ai.djl.modality.cv.Joints.Joint;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.translate.TranslatorContext;
import java.util.ArrayList;
import java.util.List;

public class SimplePoseTranslator extends ImageTranslator<Joints> {

    public SimplePoseTranslator(BaseBuilder<?> builder) {
        super(builder);
    }

    @Override
    public Joints processOutput(TranslatorContext ctx, NDList list) {
        NDArray pred = list.singletonOrThrow();
        int numJoints = (int) pred.getShape().get(1);
        int height = (int) pred.getShape().get(2);
        int width = (int) pred.getShape().get(3);
        NDArray predReshaped = pred.reshape(new Shape(1, numJoints, -1));
        NDArray maxIndices = predReshaped.argmax(2).reshape(new Shape(1, numJoints, -1));
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
                        .asType(DataType.UINT8, false)
                        .tile(2, 2)
                        .asType(DataType.BOOLEAN, false);
        float[] flattened = result.get(predMask).toFloatArray();
        float[] flattenedConfidence = maxValues.toFloatArray();
        List<Joint> joints = new ArrayList<>(numJoints);
        for (int i = 0; i < numJoints; i++) {
            joints.add(
                    new Joint(
                            flattened[i * 2] / width,
                            flattened[i * 2 + 1] / height,
                            flattenedConfidence[i]));
        }
        return new Joints(joints);
    }

    public static class Builder extends ImageTranslator.BaseBuilder<Builder> {

        @Override
        protected Builder self() {
            return this;
        }

        public SimplePoseTranslator build() {
            return new SimplePoseTranslator(this);
        }
    }
}
