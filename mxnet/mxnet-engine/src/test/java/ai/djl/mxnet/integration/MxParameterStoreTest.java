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
import ai.djl.training.tracker.Tracker;
import org.testng.Assert;
import org.testng.annotations.Test;

public class MxParameterStoreTest {

    @Test
    public void testParameterStore() {
        try (Model model = Model.newInstance("model")) {
            NDManager manager = model.getNDManager();
            int numGpus = manager.getEngine().getGpuCount();
            int numDevices;
            if (numGpus > 0) {
                numDevices = numGpus;
            } else {
                numDevices = 4;
            }
            int numWeights = Boolean.getBoolean("nightly") ? 100 : 2;
            int numUpdates = Boolean.getBoolean("nightly") ? 1000 : 10;
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
                    TestOptimizer.builder().setLearningRateTracker(Tracker.fixed(lr)).build();

            try (ParameterServer ps = new MxParameterServer(optimizer)) {

                // init
                for (int i = 0; i < numWeights; i++) {
                    ps.init(String.valueOf(i), new NDArray[] {weights[i][0]});
                }
                for (int n = 0; n < numUpdates; n++) {
                    for (int i = 0; i < numWeights; i++) {
                        ps.update(String.valueOf(i), grads[i], weights[i]);
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

        private Tracker learningRateTracker;
        int updateCount;

        protected TestOptimizer(TestOptimizer.Builder builder) {
            super(builder);
            learningRateTracker = builder.getLearningRateTracker();
        }

        /** {@inheritDoc} */
        @Override
        public void update(String parameterId, NDArray weight, NDArray grad) {
            weight.addi(
                    grad.mul(learningRateTracker.getNewValue(0))
                            .toDevice(weight.getDevice(), false));
            updateCount++;
        }

        public static Builder builder() {
            return new Builder();
        }

        public static final class Builder extends OptimizerBuilder<Builder> {

            private Tracker learningRateTracker;

            Builder() {}

            public MxParameterStoreTest.TestOptimizer.Builder setLearningRateTracker(
                    Tracker learningRateTracker) {
                this.learningRateTracker = learningRateTracker;
                return this;
            }

            public Tracker getLearningRateTracker() {
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
