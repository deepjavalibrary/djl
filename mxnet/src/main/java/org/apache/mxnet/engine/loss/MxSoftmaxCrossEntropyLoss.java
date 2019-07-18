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
package org.apache.mxnet.engine.loss;

import java.util.List;
import software.amazon.ai.Parameter;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.ndarray.types.Shape;
import software.amazon.ai.nn.core.Loss;

public class MxSoftmaxCrossEntropyLoss implements Loss {

    private int axis;
    private boolean sparseLabel;
    private boolean fromLogit;
    private float weight;
    private int batchAxis;

    public MxSoftmaxCrossEntropyLoss(
            int axis, boolean sparseLabel, boolean fromLogit, float weight, int batchAxis) {
        this.axis = axis;
        this.sparseLabel = sparseLabel;
        this.fromLogit = fromLogit;
        this.weight = weight;
        this.batchAxis = batchAxis;
    }

    @Override
    public NDArray forward(NDArray pred, NDArray label) {
        if (!fromLogit) {
            pred = pred.softmax(axis).log();
        }
        if (!sparseLabel) {
            label = label.toDense();
        }
        label = label.reshape(pred.getShape());
        NDArray loss = pred.mmul(label).sum(new int[] {axis}).mul(-weight);
        return loss.mean(new int[] {batchAxis});
    }

    @Override
    public void backward() {}

    @Override
    public boolean isInitialized() {
        return false;
    }

    @Override
    public Shape getInputShape() {
        return null;
    }

    @Override
    public Shape getOutputShape(Shape... inputs) {
        return null;
    }

    @Override
    public List<Parameter> getDirectParameters() {
        return null;
    }

    @Override
    public void beforeInitialize(NDList inputs) {}

    @Override
    public Shape getParameterShape(String name, NDList inputs) {
        return null;
    }

    @Override
    public byte[] getEncoded() {
        return new byte[0];
    }
}
