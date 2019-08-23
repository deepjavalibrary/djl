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

package org.apache.mxnet.engine;

import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.training.ParameterStore;
import software.amazon.ai.training.optimizer.Optimizer;

public class MxParameterStore implements ParameterStore {

    public MxParameterStore(Optimizer optimizer, boolean aggregateOnGPU) {
        // TODO: call KVStore JnaUtils
        createdKVStore(aggregateOnGPU);
        setOptimizer(optimizer);
    }

    /** {@inheritDoc} */
    @Override
    public void init(int key, NDArray value) {
        // TODO: call KVStore JnaUtils
    }

    /** {@inheritDoc} */
    @Override
    public void push(int key, NDArray value) {
        // TODO: call KVStore JnaUtils
    }

    /** {@inheritDoc} */
    @Override
    public void pull(int key, NDArray value) {
        // TODO: call KVStore JnaUtils
    }

    private void createdKVStore(boolean aggregateOnGPU) { // NOPMD
        // TODO: call KVStore create
    }

    private void setOptimizer(Optimizer optimizer) { // NOPMD
        // TODO: call KVStore JnaUtils
    }
}
