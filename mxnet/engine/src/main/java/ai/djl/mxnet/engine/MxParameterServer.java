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
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.training.ParameterServer;
import ai.djl.training.optimizer.Optimizer;
import com.sun.jna.Pointer;
import java.util.Arrays;

public class MxParameterServer extends NativeResource implements ParameterServer {

    public MxParameterServer(Optimizer optimizer) {
        super(createdKVStore());
        JnaUtils.parameterStoreSetUpdater(
                getHandle(), null, new OptimizerCallback(optimizer), null);
    }

    /** {@inheritDoc} */
    @Override
    public void init(String parameterId, NDArray[] values) {
        String[] keys = new String[values.length];
        Arrays.fill(keys, parameterId);
        NDList vals = new NDList(values);
        JnaUtils.parameterStoreInit(getHandle(), values.length, keys, vals);
    }

    /** {@inheritDoc} */
    @Override
    public void push(String parameterId, NDArray[] grads, int priority) {
        String[] keys = new String[grads.length];
        Arrays.fill(keys, parameterId);
        NDList vals = new NDList(grads);
        JnaUtils.parameterStorePush(getHandle(), grads.length, keys, vals, priority);
    }

    /** {@inheritDoc} */
    @Override
    public void pull(String parameterId, NDArray[] weights, int priority) {
        String[] keys = new String[weights.length];
        Arrays.fill(keys, parameterId);
        NDList vals = new NDList(weights);
        JnaUtils.parameterStorePull(getHandle(), weights.length, keys, vals, priority);
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

    private static final class OptimizerCallback implements MxnetLibrary.MXKVStoreStrUpdater {

        private Optimizer optimizer;

        OptimizerCallback(Optimizer optimizer) {
            this.optimizer = optimizer;
        }

        @Override
        public void apply(String parameterId, Pointer recv, Pointer local, Pointer handle) {
            // updater callback arguments order is: index, gradient, weight.
            MxNDManager manager = MxNDManager.getSystemManager();
            NDArray grad = manager.create(recv);
            NDArray weight = manager.create(local);
            optimizer.update(parameterId, weight, grad);
            // Above NDArrays should not be closed here.
            // TODO: avoid call JnaUtils.freeNdArray for above NDArrays
            //      and confirm these handle will be closed by MXNet engine.
        }
    }
}
