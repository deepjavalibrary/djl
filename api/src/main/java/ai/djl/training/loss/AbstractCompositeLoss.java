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
import ai.djl.util.Pair;
import java.util.List;

/**
 * {@code AbstractCompositeLoss} is a {@link Loss} class that can combine other {@link Loss}es
 * together to make a larger loss.
 *
 * <p>The AbstractCompositeLoss is designed to be extended for more complicated composite losses.
 * For simpler use cases, consider using the {@link SimpleCompositeLoss}.
 */
public abstract class AbstractCompositeLoss extends Loss {

    protected List<Loss> components;

    /**
     * Constructs a composite loss with the given name.
     *
     * @param name the display name of the loss
     */
    public AbstractCompositeLoss(String name) {
        super(name);
    }

    /**
     * Returns the inputs to computing the loss for a component loss.
     *
     * @param componentIndex the index of the component loss
     * @param labels the label input to the composite loss
     * @param predictions the predictions input to the composite loss
     * @return a pair of the (labels, predictions) inputs to the component loss
     */
    protected abstract Pair<NDList, NDList> inputForComponent(
            int componentIndex, NDList labels, NDList predictions);

    /**
     * Returns the component losses that make up the composite loss.
     *
     * @return the component losses that make up the composite loss
     */
    public List<Loss> getComponents() {
        return components;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray evaluate(NDList labels, NDList predictions) {
        NDArray[] lossComponents = new NDArray[components.size()];
        for (int i = 0; i < components.size(); i++) {
            Pair<NDList, NDList> inputs = inputForComponent(i, labels, predictions);
            lossComponents[i] = components.get(i).evaluate(inputs.getKey(), inputs.getValue());
        }
        return NDArrays.add(lossComponents);
    }

    /** {@inheritDoc} */
    @Override
    public void addAccumulator(String key) {
        for (Loss component : components) {
            component.addAccumulator(key);
        }
    }

    /** {@inheritDoc} */
    @Override
    public void updateAccumulator(String key, NDList labels, NDList predictions) {
        for (int i = 0; i < components.size(); i++) {
            Pair<NDList, NDList> inputs = inputForComponent(i, labels, predictions);
            components.get(i).updateAccumulator(key, inputs.getKey(), inputs.getValue());
        }
    }

    /** {@inheritDoc} */
    @Override
    public void resetAccumulator(String key) {
        for (Loss component : components) {
            component.resetAccumulator(key);
        }
    }

    /** {@inheritDoc} */
    @Override
    public float getAccumulator(String key) {
        return (float)
                components.stream().mapToDouble(component -> component.getAccumulator(key)).sum();
    }
}
