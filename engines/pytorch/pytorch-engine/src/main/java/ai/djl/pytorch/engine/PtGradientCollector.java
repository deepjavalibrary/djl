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

package ai.djl.pytorch.engine;

import ai.djl.ndarray.NDArray;
import ai.djl.pytorch.jni.JniUtils;
import ai.djl.training.GradientCollector;

/** {@code PtGradientCollector} is the PyTorch implementation of {@link GradientCollector}. */
public class PtGradientCollector implements GradientCollector {

    /** {@inheritDoc} */
    @Override
    public void backward(NDArray target) {
        // TODO manager should create the new NDArray on the same device
        NDArray grad =
                target.getManager()
                        .ones(target.getShape(), target.getDataType())
                        .toDevice(target.getDevice(), false);
        backward(target, grad, false, false);
    }

    /**
     * Computes the gradients of the NDArray w.r.t variables.
     *
     * @param target the target/head array to run backward on
     * @param grad The “vector” in the Jacobian-vector product, usually gradients w.r.t. each
     *     element of corresponding tensors
     * @param keepGraph whether to retain the computation graph for another backward pass on the
     *     same graph. By default the computation history is cleared.
     * @param createGraph If true, graph of the derivative will be constructed, allowing to compute
     *     higher order derivative products. Defaults to false.
     */
    private void backward(NDArray target, NDArray grad, boolean keepGraph, boolean createGraph) {
        JniUtils.backward((PtNDArray) target, (PtNDArray) grad, keepGraph, createGraph);
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        // TODO: do some clean up if necessary
    }
}
