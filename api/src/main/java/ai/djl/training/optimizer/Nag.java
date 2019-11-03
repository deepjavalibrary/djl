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
import ai.djl.training.optimizer.learningrate.LearningRateTracker;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/** An NAG optimizer. Build with {@link Nag.Builder}. */
public class Nag extends Optimizer {

    private LearningRateTracker learningRateTracker;
    private float momentum;
    private Map<String, Map<Device, NDArray>> momentumStates;

    protected Nag(Builder builder) {
        super(builder);
        learningRateTracker = builder.getLearningRateTracker();
        momentum = builder.getMomentum();
        momentumStates = new ConcurrentHashMap<>();
    }

    // TODO: make this protected after integrate with PS store
    @Override
    public void update(String parameterId, NDArray weight, NDArray grad) {
        // TODO: Support Mixed precision Sparse
        float newLearningRate = learningRateTracker.getNewLearningRate(updateCount(parameterId));
        float weightDecay = getWeightDecay(parameterId);
        NDList inputs;
        if (momentum != 0f) {
            inputs =
                    new NDList(
                            weight,
                            grad,
                            withDefaultState(
                                    momentumStates,
                                    parameterId,
                                    weight.getDevice(),
                                    k -> weight.zerosLike()));
        } else {
            inputs = new NDList(weight, grad);
        }
        NDList weights = new NDList(weight);

        NDArrayEx ex = weight.getNDArrayInternal();
        ex.nagUpdate(
                inputs, weights, newLearningRate, weightDecay, rescaleGrad, clipGrad, momentum);
    }

    public static final class Builder extends BaseBuilder<Builder> {

        private LearningRateTracker learningRateTracker;
        private float momentum;

        public Builder setLearningRateTracker(LearningRateTracker learningRateTracker) {
            this.learningRateTracker = learningRateTracker;
            return this;
        }

        public Builder setMomentum(float momentum) {
            this.momentum = momentum;
            return this;
        }

        public LearningRateTracker getLearningRateTracker() {
            return learningRateTracker;
        }

        public float getMomentum() {
            return momentum;
        }

        /** {@inheritDoc} */
        @Override
        protected Builder self() {
            return this;
        }

        public Nag build() {
            if (learningRateTracker == null) {
                throw new IllegalArgumentException("No lrTracker set");
            }
            if (momentum == 0) {
                throw new IllegalArgumentException("The momentum should be set");
            }
            return new Nag(this);
        }
    }
}
