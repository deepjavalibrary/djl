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
package ai.djl.training.optimizer;

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Function;

/** MXNet helper containing base implementations for optimizers. */
public abstract class Optimizer {

    protected float rescaleGrad;
    protected float clipGrad;
    private float weightDecays;
    private int beginNumUpdate;
    private int numUpdate;
    private Map<String, Integer> updateCounts = new ConcurrentHashMap<>();

    public Optimizer(BaseBuilder<?> builder) {
        this.rescaleGrad = builder.getRescaleGrad();
        this.weightDecays = builder.getWeightDecays();
        this.clipGrad = builder.getClipGrad();
        this.beginNumUpdate = builder.getBeginNumUpdate();

        if (rescaleGrad == 0) {
            throw new IllegalArgumentException("The rescaleGrad should be set");
        }
    }

    public static Sgd.Builder sgd() {
        return new Sgd.Builder();
    }

    public static Nag.Builder nag() {
        return new Nag.Builder();
    }

    public static Adam.Builder adam() {
        return new Adam.Builder();
    }

    protected float getWeightDecay(String parameterId) {
        return weightDecays;
    }

    protected int updateCount(String parameterId) {
        // if index exists, increment update count, if not, use begin number of update + 1
        int count =
                updateCounts.compute(
                        parameterId, (key, val) -> (val == null) ? beginNumUpdate + 1 : val + 1);
        numUpdate = Math.max(numUpdate, count);
        return numUpdate;
    }

    // TODO: make this protected after integrate with PS store
    public abstract void update(String parameterId, NDArray weight, NDArray grad);

    protected NDArray withDefaultState(
            Map<String, Map<Device, NDArray>> state,
            String key,
            Device device,
            Function<String, NDArray> defaultFunction) {
        Map<Device, NDArray> arrayMap =
                state.computeIfAbsent(
                        key,
                        k -> {
                            Map<Device, NDArray> map = new ConcurrentHashMap<>();
                            NDArray s = defaultFunction.apply(k);
                            s.detach(); // s is detached because it would be put into the optimizer
                            // callback manager and closed after the optimizer callback
                            // when using the MxParameterServer. For now, this will let it be closed
                            // by the
                            // GC when the optimizer is out of scope. Ideally, it should be put into
                            // the
                            // trainer manager instead.
                            map.put(device, s);
                            return map;
                        });
        return arrayMap.computeIfAbsent(
                device, k -> ((NDArray) arrayMap.values().toArray()[0]).asInDevice(device, true));
    }

    @SuppressWarnings("rawtypes")
    public abstract static class BaseBuilder<T extends BaseBuilder> {

        private float rescaleGrad;
        private float weightDecays;
        private float clipGrad = -1;
        private int beginNumUpdate;

        public T setRescaleGrad(float rescaleGrad) {
            this.rescaleGrad = rescaleGrad;
            return self();
        }

        public T optWeightDecays(float weightDecays) {
            this.weightDecays = weightDecays;
            return self();
        }

        public T optClipGrad(float clipGrad) {
            this.clipGrad = clipGrad;
            return self();
        }

        public T optBeginNumUpdate(int beginNumUpdate) {
            this.beginNumUpdate = beginNumUpdate;
            return self();
        }

        public float getRescaleGrad() {
            return rescaleGrad;
        }

        public float getWeightDecays() {
            return weightDecays;
        }

        public float getClipGrad() {
            return clipGrad;
        }

        public int getBeginNumUpdate() {
            return beginNumUpdate;
        }

        protected abstract T self();
    }
}
