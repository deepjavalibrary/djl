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
import java.util.ArrayList;
import java.util.List;

/**
 * {@code CompositeLoss} is an implementation of the {@link Loss} abstract class that can combine
 * different {@link Loss} functions by adding the individual losses together.
 */
public class CompositeLoss extends Loss {

    private List<Loss> components;
    private List<Integer> indices;

    /**
     * Creates a new empty instance of {@code CompositeLoss} that can combine the given {@link Loss}
     * components.
     */
    public CompositeLoss() {
        super("CompositeLoss");
        components = new ArrayList<>();
        indices = new ArrayList<>();
    }

    /**
     * Adds a Loss that applies to all labels and predictions to this composite loss.
     *
     * @param loss the loss to add
     * @return this composite loss
     */
    public Loss addLoss(Loss loss) {
        components.add(loss);
        indices.add(null);
        return this;
    }

    /**
     * Adds a Loss that applies to a single index of the label and predictions to this composite
     * loss.
     *
     * @param loss the loss to add
     * @param index the index in the label and predictions NDLists this loss applies to
     * @return this composite loss
     */
    public Loss addLoss(Loss loss, int index) {
        components.add(loss);
        indices.add(index);
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray getLoss(NDList labels, NDList predictions) {
        NDArray[] lossComponents = new NDArray[components.size()];
        for (int i = 0; i < components.size(); i++) {
            if (indices.get(i) != null) {
                int index = indices.get(i);
                lossComponents[i] =
                        components
                                .get(i)
                                .getLoss(
                                        new NDList(labels.get(index)),
                                        new NDList(predictions.get(index)));
            } else {
                lossComponents[i] = components.get(i).getLoss(labels, predictions);
            }
        }
        return NDArrays.add(lossComponents);
    }

    /** {@inheritDoc} */
    @Override
    public Loss duplicate() {
        CompositeLoss dup = new CompositeLoss();
        for (int i = 0; i < components.size(); i++) {
            if (indices.get(i) != null) {
                dup.addLoss(components.get(i).duplicate(), indices.get(i));
            } else {
                dup.addLoss(components.get(i).duplicate());
            }
        }
        return dup;
    }

    /** {@inheritDoc} */
    @Override
    public void update(NDList labels, NDList predictions) {
        for (int i = 0; i < components.size(); i++) {
            if (indices.get(i) != null) {
                int index = indices.get(i);
                components
                        .get(i)
                        .update(new NDList(labels.get(index)), new NDList(predictions.get(index)));
            } else {
                components.get(i).update(labels, predictions);
            }
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
        return (float) components.stream().mapToDouble(Loss::getValue).sum();
    }
}
