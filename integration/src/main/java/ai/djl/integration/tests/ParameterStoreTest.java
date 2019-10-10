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

package ai.djl.integration.tests;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.integration.util.Assertions;
import ai.djl.mxnet.engine.MxParameterServer;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Parameter;
import ai.djl.training.ParameterServer;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.optimizer.learningrate.LearningRateTracker;
import ai.djl.util.PairList;
import org.testng.annotations.Test;

public class ParameterStoreTest {

    @Test
    public void testParameterStore() {
        try (Model model = Model.newInstance()) {
            NDManager manager = model.getNDManager();
            int arraySize = 2;
            NDArray weight = manager.randomNormal(new Shape(5, 5)).asInDevice(Device.cpu(0), true);
            NDArray grad = manager.randomNormal(new Shape(5, 5)).asInDevice(Device.cpu(0), true);
            NDArray[] weights = {weight};
            NDArray[] grads = new NDArray[arraySize];
            for (int i = 0; i < arraySize; i++) {
                grads[i] = grad.asInDevice(Device.cpu(i), true);
            }
            float lr = .1f;
            NDArray expectedWeight = weight.add(grad.mul(arraySize).mul(lr));
            Optimizer optimizer =
                    new TestOptimizer.Builder()
                            .setRescaleGrad(1.0f)
                            .setLearningRateTracker(LearningRateTracker.fixedLearningRate(lr))
                            .build();

            try (ParameterServer ps = new MxParameterServer(optimizer)) {
                ps.init(0, weights);
                ps.push(0, grads, 0);
                ps.pull(0, weights, 0);
                Assertions.assertAlmostEquals(weights[0], expectedWeight);
            }
        }
    }

    private static class TestOptimizer extends Optimizer {
        private LearningRateTracker learningRateTracker;

        protected TestOptimizer(TestOptimizer.Builder builder) {
            super(builder);
            learningRateTracker = builder.getLearningRateTracker();
        }

        @Override
        protected boolean initializeStates(PairList<String, Parameter> parameters) {
            return true;
        }

        @Override
        public void update(int index, NDArray weight, NDArray grad) {
            weight.addi(grad.mul(learningRateTracker.getNewLearningRate(0)));
        }

        public static final class Builder extends BaseBuilder<TestOptimizer.Builder> {

            private LearningRateTracker learningRateTracker;

            public TestOptimizer.Builder setLearningRateTracker(
                    LearningRateTracker learningRateTracker) {
                this.learningRateTracker = learningRateTracker;
                return this;
            }

            public LearningRateTracker getLearningRateTracker() {
                return learningRateTracker;
            }

            @Override
            protected TestOptimizer.Builder self() {
                return this;
            }

            public TestOptimizer build() {
                if (learningRateTracker == null) {
                    throw new IllegalArgumentException("No lrTracker set");
                }
                return new TestOptimizer(this);
            }
        }
    }
}
