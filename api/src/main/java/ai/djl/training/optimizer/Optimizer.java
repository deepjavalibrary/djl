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

/**
 * An {@code Optimizer} updates the weight parameters to minimize the loss function. {@code
 * Optimizer} is an abstract class that provides the base implementation for optimizers.
 *
 * @see <a href="https://d2l.djl.ai/chapter_optimization/index.html">The D2L chapters on
 *     optimization algorithms</a>
 */
public abstract class Optimizer {

    protected float rescaleGrad;
    protected float clipGrad;
    private float weightDecays;
    private int beginNumUpdate;
    private int numUpdate;
    private Map<String, Integer> updateCounts = new ConcurrentHashMap<>();

    /**
     * Creates a new instance of {@code Optimizer}.
     *
     * @param builder the builder used to create an instance of {@code Optimizer}
     */
    public Optimizer(OptimizerBuilder<?> builder) {
        this.rescaleGrad = builder.rescaleGrad;
        this.weightDecays = builder.weightDecays;
        this.clipGrad = builder.clipGrad;
        this.beginNumUpdate = builder.beginNumUpdate;
    }

    /**
     * Returns a new instance of {@link ai.djl.training.optimizer.Sgd.Builder} that can build an
     * {@link Sgd} optimizer.
     *
     * @return the {@link Sgd} {@link ai.djl.training.optimizer.Sgd.Builder}
     */
    public static Sgd.Builder sgd() {
        return new Sgd.Builder();
    }

    /**
     * Returns a new instance of {@link ai.djl.training.optimizer.Nag.Builder} that can build an
     * {@link Nag} optimizer.
     *
     * @return the {@link Nag} {@link ai.djl.training.optimizer.Nag.Builder}
     */
    public static Nag.Builder nag() {
        return new Nag.Builder();
    }

    /**
     * Returns a new instance of {@link ai.djl.training.optimizer.Adam.Builder} that can build an
     * {@link Adam} optimizer.
     *
     * @return the {@link Adam} {@link ai.djl.training.optimizer.Adam.Builder}
     */
    public static Adam.Builder adam() {
        return new Adam.Builder();
    }

    /**
     * Returns a new instance of {@link RmsProp.Builder} that can build an {@link RmsProp}
     * optimizer.
     *
     * @return the {@link RmsProp} {@link RmsProp.Builder}
     */
    public static RmsProp.Builder rmsprop() {
        return new RmsProp.Builder();
    }

    /**
     * Returns a new instance of {@link ai.djl.training.optimizer.Adagrad.Builder} that can build an
     * {@link Adagrad} optimizer.
     *
     * @return the {@link Adagrad} {@link ai.djl.training.optimizer.Adagrad.Builder}
     */
    public static Adagrad.Builder adagrad() {
        return new Adagrad.Builder();
    }

    /**
     * Returns a new instance of {@link ai.djl.training.optimizer.Adadelta.Builder} that can build
     * an {@link Adadelta} optimizer.
     *
     * @return the {@link Adadelta} {@link ai.djl.training.optimizer.Adadelta.Builder}
     */
    public static Adadelta.Builder adadelta() {
        return new Adadelta.Builder();
    }

    /**
     * Gets the value of weight decay.
     *
     * @return the value of weight decay
     */
    protected float getWeightDecay() {
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

    /**
     * Updates the parameters according to the gradients.
     *
     * @param parameterId the parameter to be updated
     * @param weight the weights of the parameter
     * @param grad the gradients
     */
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
                            // TODO attach s to the NDManager of ParameterStore
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
                device, k -> arrayMap.values().iterator().next().toDevice(device, true));
    }

    /** The Builder to construct an {@link Optimizer}. */
    @SuppressWarnings("rawtypes")
    public abstract static class OptimizerBuilder<T extends OptimizerBuilder> {

        private float rescaleGrad = 1.0f;
        private float weightDecays;
        private float clipGrad = -1;
        private int beginNumUpdate;

        protected OptimizerBuilder() {}

        /**
         * Sets the value used to rescale the gradient. This is used to alleviate the effect of
         * batching on the loss. Usually, the value is set to \( 1/batch_size \). Defaults to 1.
         *
         * @param rescaleGrad the value used to rescale the gradient
         * @return this {@code Builder}
         */
        public T setRescaleGrad(float rescaleGrad) {
            this.rescaleGrad = rescaleGrad;
            return self();
        }

        /**
         * Sets the value of weight decay. Weight decay augments the objective function with a
         * regularization term that penalizes large weights.
         *
         * @param weightDecays the value of weight decay to be set
         * @return this {@code Builder}
         */
        public T optWeightDecays(float weightDecays) {
            this.weightDecays = weightDecays;
            return self();
        }

        /**
         * Sets the value of the \(clipGrad\). Clips the gradient to the range of \([-clipGrad,
         * clipGrad]\). If \(clipGrad \lt 0\), gradient clipping is turned off.
         *
         * <p>\(grad = max(min(grad, clipGrad), -clipGrad)\)
         *
         * @param clipGrad the value of \(clipGrad\)
         * @return this {@code Builder}
         */
        public T optClipGrad(float clipGrad) {
            this.clipGrad = clipGrad;
            return self();
        }

        /**
         * Sets the initial value of the number of updates.
         *
         * @param beginNumUpdate the initial value of the number of updates
         * @return this {@code Builder}
         */
        public T optBeginNumUpdate(int beginNumUpdate) {
            this.beginNumUpdate = beginNumUpdate;
            return self();
        }

        protected abstract T self();
    }
}
