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

/**
 * {@code CompositeLoss} is an implementation of the {@link Loss} abstract class that can combine
 * different {@link Loss} functions by adding the individual losses together.
 */
public class CompositeLoss extends Loss {

    private Loss[] components;

    /**
     * Creates a new instance of {@code CompositeLoss} that can combine the given {@link Loss}
     * components.
     *
     * @param components the {@code Loss} objects that form the composite loss
     */
    public CompositeLoss(Loss... components) {
        super("CompositeLoss");
        this.components = components;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray getLoss(NDList label, NDList prediction) {
        return NDArrays.add(
                Arrays.stream(components)
                        .map(component -> component.getLoss(label, prediction))
                        .toArray(NDArray[]::new));
    }

    /** {@inheritDoc} */
    @Override
    public Loss duplicate() {
        return new CompositeLoss(
                Arrays.stream(components).map(Loss::duplicate).toArray(Loss[]::new));
    }

    /** {@inheritDoc} */
    @Override
    public void update(NDList labels, NDList predictions) {
        for (Loss component : components) {
            component.update(labels, predictions);
        }
    }

    /** {@inheritDoc} */
    @Override
    public void reset() {
        for (Loss component : components) {
            component.reset();
        }
    }

    /** {@inheritDoc} */
    @Override
    public float getValue() {
        return (float) Arrays.stream(components).mapToDouble(Loss::getValue).sum();
    }
}
