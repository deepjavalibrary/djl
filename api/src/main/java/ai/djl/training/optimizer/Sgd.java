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

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.internal.NDArrayEx;
import ai.djl.nn.Parameter;
import ai.djl.training.optimizer.learningrate.LearningRateTracker;
import ai.djl.util.PairList;
import java.util.ArrayList;
import java.util.List;

/** An SGD optimizer. Build with {@link Sgd.Builder}. */
public class Sgd extends Optimizer {

    private LearningRateTracker learningRateTracker;
    private float momentum;
    private boolean lazyUpdate;
    private List<NDArray> momentumStates;

    protected Sgd(Builder builder) {
        super(builder);
        learningRateTracker = builder.getLearningRateTracker();
        momentum = builder.getMomentum();
        lazyUpdate = builder.isLazyUpdate();
    }

    @Override
    protected boolean initializeStates(PairList<String, Parameter> parameters) {
        if (momentum != 0f) {
            momentumStates = new ArrayList<>(parameters.size());
            for (Parameter param : parameters.values()) {
                momentumStates.add(param.getArray().zerosLike());
            }
        }
        return true;
    }

    // TODO: make this protected after integrate with PS store
    @Override
    public void update(int index, NDArray weight, NDArray grad) {
        // TODO: Support Mixed precision Sparse
        float weightDecay = getWeightDecay(index);
        float learningRate = learningRateTracker.getNewLearningRate(updateCount(index));
        NDList inputs;
        // TODO: check momentum correctness
        if (momentum != 0f) {
            inputs = new NDList(weight, grad, momentumStates.get(index));
        } else {
            inputs = new NDList(weight, grad);
        }
        NDList weights = new NDList(weight);

        NDArrayEx ex = weight.getNDArrayInternal();
        ex.sgdUpdate(
                inputs,
                weights,
                learningRate,
                weightDecay,
                rescaleGrad,
                clipGrad,
                momentum,
                lazyUpdate);
    }

    public static final class Builder extends BaseBuilder<Builder> {

        private LearningRateTracker learningRateTracker;
        private float momentum;
        private boolean lazyUpdate = true;

        @Override
        protected Builder self() {
            return this;
        }

        public Builder setLearningRateTracker(LearningRateTracker learningRateTracker) {
            this.learningRateTracker = learningRateTracker;
            return this;
        }

        public Builder optMomentum(float momentum) {
            this.momentum = momentum;
            return this;
        }

        public Builder optLazyUpdate(boolean lazyUpdate) {
            this.lazyUpdate = lazyUpdate;
            return this;
        }

        public LearningRateTracker getLearningRateTracker() {
            return learningRateTracker;
        }

        public float getMomentum() {
            return momentum;
        }

        public boolean isLazyUpdate() {
            return lazyUpdate;
        }

        public Sgd build() {
            if (learningRateTracker == null) {
                throw new IllegalArgumentException("No lrTracker set");
            }
            return new Sgd(this);
        }
    }
}
