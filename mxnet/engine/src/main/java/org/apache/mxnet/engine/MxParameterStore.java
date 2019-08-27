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

import com.sun.jna.Pointer;
import org.apache.mxnet.jna.JnaUtils;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.training.ParameterStore;
import software.amazon.ai.training.optimizer.Optimizer;

public class MxParameterStore extends NativeResource implements ParameterStore {

    public MxParameterStore(boolean aggregateOnGPU, Optimizer optimizer) {
        super(createdKVStore(aggregateOnGPU));
        setOptimizer(optimizer);
    }

    /** {@inheritDoc} */
    @Override
    public void init(int key, NDArray value) {
        // TODO: handle list
        int[] keys = {key};
        NDList vals = new NDList(value);
        JnaUtils.parameterStoreInit(getHandle(), 1, keys, vals);
    }

    /** {@inheritDoc} */
    @Override
    public void push(int key, NDArray value) {
        // TODO: handle list
        int[] keys = {key};
        NDList vals = new NDList(value);
        JnaUtils.parameterStorePush(getHandle(), 1, keys, vals, 0);
    }

    /** {@inheritDoc} */
    @Override
    public void pull(int key, NDArray value) {
        // TODO: handle list
        int[] keys = {key};
        NDList vals = new NDList(value);
        JnaUtils.parameterStorePull(getHandle(), 1, keys, vals, 0);
    }

    private static Pointer createdKVStore(boolean aggregateOnGPU) {
        Pointer handle;
        if (aggregateOnGPU) {
            handle = JnaUtils.parameterStoreCreate("device");
        } else {
            handle = JnaUtils.parameterStoreCreate("local");
        }
        return handle;
    }

    private void setOptimizer(Optimizer optimizer) { // NOPMD
        // TODO: call KVStore JnaUtils
    }

    @Override
    public void close() {
        Pointer pointer = handle.getAndSet(null);
        if (pointer != null) {
            JnaUtils.parameterStoreClose(pointer);
        }
    }
}
