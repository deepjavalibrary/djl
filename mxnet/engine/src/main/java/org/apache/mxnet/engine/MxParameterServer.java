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
import java.util.Arrays;
import org.apache.mxnet.jna.JnaUtils;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.training.ParameterServer;
import software.amazon.ai.training.optimizer.Optimizer;

public class MxParameterServer extends NativeResource implements ParameterServer {

    public MxParameterServer(Optimizer optimizer) {
        super(createdKVStore());
        setOptimizer(optimizer);
    }

    /** {@inheritDoc} */
    @Override
    public void init(int key, NDArray[] values) {
        // We are suppoting a single key on multiple devices right now
        // Duplicate keys, to length of values, may need to change in future
        int[] keys = new int[values.length];
        Arrays.fill(keys, key);
        NDList vals = new NDList(values);
        JnaUtils.parameterStoreInit(getHandle(), values.length, keys, vals);
    }

    /** {@inheritDoc} */
    @Override
    public void push(int key, NDArray[] values) {
        int[] keys = new int[values.length];
        Arrays.fill(keys, key);
        NDList vals = new NDList(values);
        JnaUtils.parameterStorePush(getHandle(), values.length, keys, vals, 0);
    }

    /** {@inheritDoc} */
    @Override
    public void pull(int key, NDArray[] values) {
        int[] keys = new int[values.length];
        Arrays.fill(keys, key);
        NDList vals = new NDList(values);
        JnaUtils.parameterStorePull(getHandle(), values.length, keys, vals, 0);
    }

    private static Pointer createdKVStore() {
        return createdKVStore(true);
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
