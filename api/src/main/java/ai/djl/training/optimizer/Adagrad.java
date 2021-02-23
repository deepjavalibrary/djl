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
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.internal.NDArrayEx;
import ai.djl.ndarray.types.SparseFormat;
import ai.djl.training.tracker.Tracker;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * {@code Adagrad} is an AdaGrad {@link Optimizer}.
 *
 * <p>This class implements
 *
 * <p>Adagrad updates the weights using:<br>
 * <br>
 * \( grad = clip(grad * resc_grad, clip_grad) + wd * weight \)<br>
 * \( history += grad^2 \)<br>
 * \( weight -= lr * grad / (sqrt(history) + epsilon) \)<br>
 * <br>
 * where grad represents the gradient, wd represents weight decay, and lr represents learning rate.
 *
 * @see <a href="https://d2l.djl.ai/chapter_optimization/adagrad.html">The D2L chapter on
 *     Adagrad</a>
 */
public class Adagrad extends Optimizer {

    private Tracker learningRateTracker;
    private float epsilon;

    private Map<String, Map<Device, NDArray>> history;

    /**
     * Creates a new instance of {@code Adam} optimizer.
     *
     * @param builder the builder to create a new instance of {@code Adam} optimizer
     */
    protected Adagrad(Builder builder) {
        super(builder);
        learningRateTracker = builder.learningRateTracker;
        epsilon = builder.epsilon;
        history = new ConcurrentHashMap<>();
    }

    /** {@inheritDoc} */
    @Override
    public void update(String parameterId, NDArray weight, NDArray grad) {
        int t = updateCount(parameterId);
        float newLearningRate = learningRateTracker.getNewValue(t);
        float weightDecay = getWeightDecay();

        if (Float.isNaN(newLearningRate)
                || Float.isNaN(weightDecay)
                || Float.isInfinite(newLearningRate)
                || Float.isInfinite(weightDecay)) {
            throw new IllegalStateException("learning rate or weight decay is nan or infinite");
        }
        NDList inputs =
                new NDList(
                        weight,
                        grad.toSparse(SparseFormat.ROW_SPARSE), // FIXME: add regular adagrad MxNet
                        withDefaultState(
                                history, parameterId, weight.getDevice(), k -> weight.zerosLike()));

        NDList weights = new NDList(weight);

        NDArrayEx ex = weight.getNDArrayInternal();

        // TODO: change to our own implementation
        ex.adagradUpdate(
                inputs, weights, newLearningRate, weightDecay, rescaleGrad, clipGrad, epsilon);
    }

    /**
     * Creates a builder to build a {@code Adam}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /** The Builder to construct an {@link Adagrad} object. */
    public static final class Builder extends OptimizerBuilder<Builder> {

        private Tracker learningRateTracker = Tracker.fixed(0.001f);
        private float epsilon = 1e-8f;

        Builder() {}

        /** {@inheritDoc} */
        @Override
        protected Builder self() {
            return this;
        }

        /**
         * Sets the {@link Tracker} for this optimizer.
         *
         * @param learningRateTracker the {@link Tracker} to be set
         * @return this {@code Builder}
         */
        public Builder optLearningRateTracker(Tracker learningRateTracker) {
            this.learningRateTracker = learningRateTracker;
            return this;
        }

        /**
         * Sets \(epsilon\) - a small quantity for numerical stability.
         *
         * @param epsilon a small quantity for numerical stability
         * @return this {@code Builder}
         */
        public Builder optEpsilon(float epsilon) {
            this.epsilon = epsilon;
            return this;
        }

        /**
         * Builds a {@link Adagrad} block.
         *
         * @return the {@link Adagrad} block
         */
        public Adagrad build() {
            return new Adagrad(this);
        }
    }
}
