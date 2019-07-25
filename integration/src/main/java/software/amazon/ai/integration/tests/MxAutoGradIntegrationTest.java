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

import java.util.List;
import java.util.function.Function;
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
import software.amazon.ai.ndarray.types.Shape;
import software.amazon.ai.nn.core.Linear;
import software.amazon.ai.training.Loss;
import software.amazon.ai.training.initializer.Initializer;
import software.amazon.ai.util.PairList;
import software.amazon.ai.util.RandomUtils;

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
            int epochs = 5;

            PairList<Double, Double> pairList = randomGen(100, 1000, 3, numOfData, Math::sqrt);

            Linear block = new Linear.Builder().setOutChannels(1).build();
            block.setInitializer(manager, Initializer.ONES);

            MxOptimizer optimizer =
                    new Sgd(1.0f, 0.f, 0.f, MxLearningRateTracker.fixedLR(0.01f), 0, 0.f, true);

            for (int epoch = 0; epoch < epochs; epoch++) {
                for (int i = 0; i < numOfData; i++) {
                    MxAutograd autograd = new MxAutograd();
                    NDArray x = manager.create(new double[] {pairList.keyAt(i)});
                    autograd.attachGradient(x);
                    NDArray pred = block.forward(x);
                    NDArray label = manager.create(new double[] {pairList.valueAt(i)});
                    NDArray loss = Loss.l2Loss(pred, label, 1, 0);
                    autograd.backward((MxNDArray) loss);
                    List<Parameter> params = block.getParameters();
                    for (Parameter param : params) {
                        NDArray weight = param.getArray();
                        NDArray grad = autograd.getGradient((MxNDArray) x);
                        optimizer.update(0, weight, grad.reshape(weight.getShape()), null);
                    }
                    autograd.close();
                    System.out.println("Epoch " + (epoch + 1) + " loss " + loss); // NOPMD
                }
            }
        }
    }

    private PairList<Double, Double> randomGen(
            int min, int max, int noiseScale, int num, Function<Double, Double> function) {
        PairList<Double, Double> pairList = new PairList<>();
        for (int i = 0; i < num; i++) {
            double x = (Math.random() * ((max - min) + 1)) + min;
            double y = function.apply(x);
            y = y + RandomUtils.nextGaussian() * noiseScale;
            pairList.add(x, y);
        }
        return pairList;
    }
}
