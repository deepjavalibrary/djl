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
package ai.djl.training.loss;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.training.Activation;

public class SigmoidBinaryCrossEntropyLoss extends Loss {

    private float weight;
    private int batchAxis;
    private boolean fromSigmoid;

    /**
     * The Sigmoid cross-entropy loss for binary classification.
     *
     * @param weight weight to apply on loss value, default 1
     * @param batchAxis axis that represents mini-batch, default 0
     * @param fromSigmoid Whether the input is from the output of sigmoid, default false
     */
    public SigmoidBinaryCrossEntropyLoss(float weight, int batchAxis, boolean fromSigmoid) {
        this.weight = weight;
        this.batchAxis = batchAxis;
        this.fromSigmoid = fromSigmoid;
    }

    /** The Sigmoid cross-entropy loss for binary classification. */
    public SigmoidBinaryCrossEntropyLoss() {
        weight = 1;
        batchAxis = 0;
        fromSigmoid = false;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray getLoss(NDArray label, NDArray prediction) {
        label = label.reshape(prediction.getShape());
        NDArray loss;
        if (!fromSigmoid) {
            // TODO: Add Position weight option
            loss =
                    Activation.relu(prediction)
                            .sub(prediction.mul(label))
                            .add(Activation.softrelu(prediction.abs().neg()));
        } else {
            double eps = 1e-12;
            loss =
                    prediction
                            .add(eps)
                            .log()
                            .mul(label)
                            .add(
                                    NDArrays.sub(1., prediction)
                                            .add(eps)
                                            .mul(NDArrays.sub(1., label)));
        }
        if (weight != 1f) {
            loss = loss.mul(weight);
        }
        return loss.mean(excludeBatchAxis(loss, batchAxis));
    }
}
