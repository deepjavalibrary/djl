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
package software.amazon.ai.integration.tests;

import java.util.Collection;
import org.apache.mxnet.engine.MxAutograd;
import org.apache.mxnet.engine.MxNDArray;
import org.apache.mxnet.engine.lrscheduler.MxLearningRateTracker;
import org.apache.mxnet.engine.optimizer.MxOptimizer;
import org.apache.mxnet.engine.optimizer.Sgd;
import software.amazon.ai.Parameter;
import software.amazon.ai.integration.exceptions.FailedTestException;
import software.amazon.ai.integration.util.AbstractTest;
import software.amazon.ai.integration.util.Assertions;
import software.amazon.ai.integration.util.RunAsTest;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDArrays;
import software.amazon.ai.ndarray.NDManager;
import software.amazon.ai.ndarray.index.NDIndex;
import software.amazon.ai.ndarray.types.DataType;
import software.amazon.ai.ndarray.types.Shape;
import software.amazon.ai.nn.core.Linear;
import software.amazon.ai.training.Loss;
import software.amazon.ai.training.initializer.Initializer;

public class MxAutoGradIntegrationTest extends AbstractTest {

    // TODO use API level integration test once moved Autograd to API package
    public static void main(String[] args) {
        new MxAutoGradIntegrationTest().runTest(args);
    }

    @RunAsTest
    public void testAutograd() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager();
                MxAutograd autograd = new MxAutograd()) {
            NDArray lhs = manager.create(new float[] {6, -9, -12, 15, 0, 4}, new Shape(2, 3));
            NDArray rhs = manager.create(new float[] {2, 3, -4}, new Shape(3, 1));
            autograd.attachGradient(lhs);
            // autograd automatically set recording and training during initialization
            Assertions.assertTrue(MxAutograd.isRecording());
            Assertions.assertTrue(MxAutograd.isTraining());
            NDArray result = NDArrays.mmul(lhs, rhs);
            autograd.backward((MxNDArray) result);
        }
    }

    @RunAsTest
    public void testTrain() {
        try (NDManager manager = NDManager.newBaseManager()) {
            int numOfData = 1000;
            int batchSize = 10;
            int epochs = 10;

            NDArray weight = manager.create(new float[] {2.f, -3.4f}, new Shape(2, 1));
            float bias = 4.2f;
            NDArray data = manager.randomNormal(new Shape(numOfData, weight.size(0)));
            // y = w * x + b
            NDArray label = data.mmul(weight).add(bias);
            // add noise
            label.add(
                    manager.randomNormal(
                            0, 0.01, label.getShape(), DataType.FLOAT32, manager.getContext()));
            Linear block = new Linear.Builder().setOutChannels(1).build();
            block.setInitializer(manager, Initializer.ONES);

            MxOptimizer optimizer =
                    new Sgd(
                            1.0f / batchSize,
                            0.f,
                            -1,
                            MxLearningRateTracker.fixedLR(0.03f),
                            0,
                            0.f,
                            true);
            NDArray loss = manager.create(0.f);

            for (int epoch = 0; epoch < epochs; epoch++) {
                for (int i = 0; i < numOfData / batchSize; i++) {
                    try (MxAutograd autograd = new MxAutograd()) {
                        NDIndex indices = new NDIndex(i * batchSize + ":" + batchSize * (i + 1));
                        NDArray x = data.get(indices);
                        NDArray y = label.get(indices);
                        NDArray yHat = block.forward(x);
                        loss = Loss.l2Loss(yHat, y, 1, 0);
                        autograd.backward((MxNDArray) loss);
                    }
                    Collection<Parameter> params = block.getParameters().values();
                    for (Parameter param : params) {
                        NDArray paramArray = param.getArray();
                        NDArray grad = paramArray.getGradient();
                        optimizer.update(0, paramArray, grad, null);
                    }
                }
            }
            assert loss.toFloatArray()[0] < 0.001f;
        }
    }
}
