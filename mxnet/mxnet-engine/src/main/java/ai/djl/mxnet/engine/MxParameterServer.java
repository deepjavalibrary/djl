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
import ai.djl.ndarray.NDManager;
import ai.djl.training.ParameterServer;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.util.NativeResource;
import com.sun.jna.Pointer;
import java.util.Arrays;

/** {@code MxParameterServer} is the MXNet implementation of {@link ParameterServer}. */
public class MxParameterServer extends NativeResource<Pointer> implements ParameterServer {

    @SuppressWarnings("PMD.SingularField")
    // use class field to hold the OptimizerCallback which prevent it from being gc.
    private OptimizerCallback callback;

    private int priority;

    /**
     * Constructs a new {@code MxParameterServer}.
     *
     * @param optimizer the optimizer to use for the parameter server updates
     */
    public MxParameterServer(Optimizer optimizer) {
        super(createdKVStore());
        callback = new OptimizerCallback(optimizer);
        JnaUtils.parameterStoreSetUpdater(getHandle(), null, callback, null);
        priority = 0;
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
    public void update(String parameterId, NDArray[] grads, NDArray[] params) {
        String[] gradKeys = new String[grads.length];
        String[] paramKeys = new String[params.length];
        Arrays.fill(gradKeys, parameterId);
        Arrays.fill(paramKeys, parameterId);
        JnaUtils.parameterStorePushPull(
                getHandle(),
                grads.length,
                gradKeys,
                params.length,
                paramKeys,
                new NDList(grads),
                new NDList(params),
                -priority);
        priority++;
    }

    private static Pointer createdKVStore() {
        return JnaUtils.parameterStoreCreate("device");
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        Pointer pointer = handle.getAndSet(null);
        if (pointer != null) {
            JnaUtils.parameterStoreClose(pointer);
        }
    }

    /** A helper to wrap the optimizer so it can be called by the MXNet KVStore. */
    private static final class OptimizerCallback implements MxnetLibrary.MXKVStoreStrUpdater {

        private Optimizer optimizer;

        OptimizerCallback(Optimizer optimizer) {
            this.optimizer = optimizer;
        }

        /** {@inheritDoc} */
        @Override
        public void apply(String parameterId, Pointer recv, Pointer local, Pointer handle) {
            // updater callback arguments order is: index, gradient, weight.
            try (NDManager manager = MxNDManager.getSystemManager().newSubManager()) {
                MxNDManager m = (MxNDManager) manager;
                MxNDArray grad = m.create(recv);
                MxNDArray weight = m.create(local);
                optimizer.update(parameterId, weight, grad);
            }
        }
    }
}
