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

package ai.djl.mxnet.engine;

import ai.djl.mxnet.jna.JnaUtils;
import ai.djl.mxnet.jna.MxnetLibrary;
import com.sun.jna.Pointer;
import java.util.Arrays;
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
    public void push(int key, NDArray[] values, int priority) {
        int[] keys = new int[values.length];
        Arrays.fill(keys, key);
        NDList vals = new NDList(values);
        JnaUtils.parameterStorePush(getHandle(), values.length, keys, vals, priority);
    }

    /** {@inheritDoc} */
    @Override
    public void pull(int key, NDArray[] values, int priority) {
        int[] keys = new int[values.length];
        Arrays.fill(keys, key);
        NDList vals = new NDList(values);
        JnaUtils.parameterStorePull(getHandle(), values.length, keys, vals, priority);
    }

    final void setOptimizer(Optimizer optimizer) {
        JnaUtils.parameterStoreSetUpdater(getHandle(), new OptimizerCallback(optimizer), null);
    }

    private static Pointer createdKVStore() {
        return JnaUtils.parameterStoreCreate("device");
    }

    @Override
    public void close() {
        Pointer pointer = handle.getAndSet(null);
        if (pointer != null) {
            JnaUtils.parameterStoreClose(pointer);
        }
    }

    private static class OptimizerCallback implements MxnetLibrary.MXKVStoreUpdater {
        private Optimizer optimizer;

        OptimizerCallback(Optimizer optimizer) {
            this.optimizer = optimizer;
        }

        @Override
        public void apply(int key, Pointer recv, Pointer local, Pointer handle) {
            // updater callback arguments order is: index, gradient, weight.
            NDArray grad = MxNDManager.getSystemManager().create(local);
            NDArray weight = MxNDManager.getSystemManager().create(recv);
            optimizer.update(key, grad, weight);
            grad.close();
            weight.close();
        }
    }
}
