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

package ai.djl.mxnet.integration;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.mxnet.engine.MxParameterServer;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.testing.Assertions;
import ai.djl.training.ParameterServer;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.optimizer.learningrate.LearningRateTracker;
import org.testng.Assert;
import org.testng.annotations.Test;

public class MxParameterStoreTest {

    @Test
    public void testParameterStore() {
        try (Model model = Model.newInstance("model")) {
            NDManager manager = model.getNDManager();
            int numGpus = Device.getGpuCount();
            int numDevices;
            if (numGpus > 0) {
                numDevices = numGpus;
            } else {
                numDevices = 4;
            }
            int numWeights = 2;
            // TODO: this test is currently flaky with large numUpdates and large numWeights
            int numUpdates = 10;
            NDArray[][] weights = new NDArray[numWeights][numDevices];
            NDArray[][] grads = new NDArray[numWeights][numDevices];
            NDArray[] expected = new NDArray[numWeights];
            float lr = .1f;
            for (int i = 0; i < numWeights; i++) {
                NDArray w = manager.randomNormal(new Shape(1, 1));
                NDArray g = manager.randomNormal(new Shape(1, 1));
                // simulate aggregate gradient from all device and apply on weight
                expected[i] = w;
                for (int n = 0; n < numUpdates; n++) {
                    expected[i] = updateHelper(expected[i], g, numDevices, lr);
                }
                // copy weight and gradient to all devices
                for (int j = 0; j < numDevices; j++) {
                    Device device;
                    if (numGpus > 0) {
                        device = Device.gpu(j);
                    } else {
                        device = Device.cpu();
                    }
                    weights[i][j] = w.toDevice(device, true);
                    grads[i][j] = g.toDevice(device, true);
                }
            }

            TestOptimizer optimizer =
                    TestOptimizer.builder()
                            .setLearningRateTracker(LearningRateTracker.fixedLearningRate(lr))
                            .build();

            try (ParameterServer ps = new MxParameterServer(optimizer)) {

                // init
                for (int i = 0; i < numWeights; i++) {
                    ps.init(String.valueOf(i), new NDArray[] {weights[i][0]});
                }
                for (int n = 0; n < numUpdates; n++) {
                    // push
                    for (int i = 0; i < numWeights; i++) {
                        ps.push(String.valueOf(i), grads[i], -i);
                    }
                    // pull
                    for (int i = 0; i < numWeights; i++) {
                        ps.pull(String.valueOf(i), weights[i], -i);
                    }
                }
                for (int i = 0; i < numWeights; i++) {
                    Assertions.assertAlmostEquals(weights[i][0], expected[i]);
                    // check the number of updates has been invoked
                    Assert.assertEquals(optimizer.updateCount, numWeights * numUpdates);
                }
            }
        }
    }

    private static NDArray updateHelper(NDArray weight, NDArray grad, int numDevices, float lr) {
        return weight.add(grad.mul(numDevices).mul(lr));
    }

    private static class TestOptimizer extends Optimizer {

        private LearningRateTracker learningRateTracker;
        int updateCount;

        protected TestOptimizer(TestOptimizer.Builder builder) {
            super(builder);
            learningRateTracker = builder.getLearningRateTracker();
        }

        /** {@inheritDoc} */
        @Override
        public void update(String parameterId, NDArray weight, NDArray grad) {
            weight.addi(
                    grad.mul(learningRateTracker.getNewLearningRate(0))
                            .toDevice(weight.getDevice(), false));
            updateCount++;
        }

        public static Builder builder() {
            return new Builder();
        }

        public static final class Builder extends OptimizerBuilder<Builder> {

            private LearningRateTracker learningRateTracker;

            Builder() {}

            public MxParameterStoreTest.TestOptimizer.Builder setLearningRateTracker(
                    LearningRateTracker learningRateTracker) {
                this.learningRateTracker = learningRateTracker;
                return this;
            }

            public LearningRateTracker getLearningRateTracker() {
                return learningRateTracker;
            }

            /** {@inheritDoc} */
            @Override
            protected MxParameterStoreTest.TestOptimizer.Builder self() {
                return this;
            }

            public MxParameterStoreTest.TestOptimizer build() {
                if (learningRateTracker == null) {
                    throw new IllegalArgumentException("No lrTracker set");
                }
                return new MxParameterStoreTest.TestOptimizer(this);
            }
        }
    }
}
