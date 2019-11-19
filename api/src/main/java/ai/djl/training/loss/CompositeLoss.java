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
import ai.djl.ndarray.NDList;
import java.util.Arrays;

public class CompositeLoss extends Loss {

    private Loss[] components;

    public CompositeLoss(Loss... components) {
        super("CompositeLoss");
        this.components = components;
    }

    @Override
    public NDArray getLoss(NDList label, NDList prediction) {
        NDArray[] losses =
                Arrays.stream(components)
                        .map(component -> component.getLoss(label, prediction))
                        .toArray(NDArray[]::new);
        try (NDArray loss = NDArrays.stack(new NDList(losses))) {
            return loss.sum(new int[] {0});
        }
    }

    @Override
    public Loss duplicate() {
        return new CompositeLoss(
                Arrays.stream(components).map(Loss::duplicate).toArray(Loss[]::new));
    }

    @Override
    public void update(NDList labels, NDList predictions) {
        for (Loss component : components) {
            component.update(labels, predictions);
        }
    }

    @Override
    public void reset() {
        for (Loss component : components) {
            component.reset();
        }
    }

    @Override
    public float getValue() {
        return (float) Arrays.stream(components).mapToDouble(Loss::getValue).sum();
    }
}
