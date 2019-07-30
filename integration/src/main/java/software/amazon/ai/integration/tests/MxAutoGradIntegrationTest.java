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

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import org.apache.mxnet.engine.MxAutograd;
import org.apache.mxnet.engine.MxNDArray;
import org.apache.mxnet.engine.lrscheduler.MxLearningRateTracker;
import org.apache.mxnet.engine.optimizer.MxOptimizer;
import org.apache.mxnet.engine.optimizer.Sgd;
import software.amazon.ai.Block;
import software.amazon.ai.Parameter;
import software.amazon.ai.integration.IntegrationTest;
import software.amazon.ai.integration.exceptions.FailedTestException;
import software.amazon.ai.integration.util.Assertions;
import software.amazon.ai.integration.util.RunAsTest;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDArrays;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.ndarray.NDManager;
import software.amazon.ai.ndarray.index.NDIndex;
import software.amazon.ai.ndarray.types.DataType;
import software.amazon.ai.ndarray.types.Shape;
import software.amazon.ai.nn.core.Linear;
import software.amazon.ai.training.Activation;
import software.amazon.ai.training.Loss;
import software.amazon.ai.training.initializer.Initializer;
import software.amazon.ai.training.initializer.NormalInitializer;
import software.amazon.ai.training.metrics.Accuracy;
import software.amazon.ai.util.PairList;

public class MxAutoGradIntegrationTest {

    public static void main(String[] args) {
        String[] cmd = new String[] {"-c", MxAutoGradIntegrationTest.class.getName()};
        new IntegrationTest().runTests(cmd);
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

    // @RunAsTest
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

    // @RunAsTest
    public void testTrainMnist() throws IOException {

        class Mlp implements Block {

            private Linear fc1;
            private Linear fc2;
            private Linear fc3;

            public Mlp() {
                fc1 = new Linear.Builder().setOutChannels(128).build();
                fc2 = new Linear.Builder().setOutChannels(64).build();
                fc3 = new Linear.Builder().setOutChannels(10).build();
            }

            @Override
            public NDList forward(NDList inputs, PairList<String, String> params) {
                NDArray data = inputs.head();
                NDArray fc1Nd = fc1.forward(data);
                NDArray relu1Nd = Activation.relu(fc1Nd);
                NDArray fc2Nd = fc2.forward(relu1Nd);
                NDArray relu2Nd = Activation.relu(fc2Nd);
                NDArray fc3Nd = fc3.forward(relu2Nd);
                return new NDList(fc3Nd);
            }

            @Override
            public Map<String, Block> getChildren() {
                Map<String, Block> map = new ConcurrentHashMap<>();
                map.put("fc1", fc1);
                map.put("fc2", fc2);
                map.put("fc3", fc3);
                return map;
            }

            @Override
            public void backward() {}

            @Override
            public boolean isInitialized() {
                return false;
            }

            @Override
            public Shape getInputShape() {
                return null;
            }

            @Override
            public Shape getOutputShape(Shape... inputs) {
                return null;
            }

            @Override
            public List<Parameter> getDirectParameters() {
                return new ArrayList<>();
            }

            @Override
            public void beforeInitialize(NDList inputs) {}

            @Override
            public Shape getParameterShape(String name, NDList inputs) {
                return null;
            }

            @Override
            public byte[] getEncoded() {
                return new byte[0];
            }
        }

        try (NDManager manager = NDManager.newBaseManager()) {

            // URL to Download: http://data.mxnet.io/mxnet/data/mnist.zip

            // TODO: Remove this line with DataLoader
            byte[] imageBytesRaw =
                    Files.readAllBytes(
                            Paths.get(
                                    "/Users/qingla/testEnv/testMX/mnist/train-images-idx3-ubyte"));
            byte[] labelBytesRaw =
                    Files.readAllBytes(
                            Paths.get(
                                    "/Users/qingla/testEnv/testMX/mnist/train-labels-idx1-ubyte"));

            byte[] imageBytes = new byte[imageBytesRaw.length - 16];
            System.arraycopy(imageBytesRaw, 16, imageBytes, 0, imageBytes.length);
            byte[] labelBytes = new byte[labelBytesRaw.length - 8];
            System.arraycopy(labelBytesRaw, 8, labelBytes, 0, labelBytes.length);

            NDArray data = manager.create(new Shape(labelBytes.length, 28, 28), DataType.UINT8);
            data.set(imageBytes);
            data = data.asType(DataType.FLOAT32, true);

            NDArray label = manager.create(new Shape(labelBytes.length), DataType.UINT8);
            label.set(labelBytes);
            label = label.asType(DataType.FLOAT32, true);

            Block mlp = new Mlp();
            mlp.setInitializer(manager, new NormalInitializer(0.01));

            int numEpoch = 10;
            int batchSize = 100;
            int numBatches = 60000 / batchSize;

            MxOptimizer optimizer =
                    new Sgd(
                            1.0f / batchSize,
                            0.f,
                            -1f,
                            MxLearningRateTracker.fixedLR(0.1f),
                            0,
                            0.9f,
                            true);

            for (int epoch = 0; epoch < numEpoch; epoch++) {
                String lossString = "";
                Accuracy acc = new Accuracy();
                for (int i = 0; i < numBatches; i++) {
                    String expression = i * batchSize + ":" + (i + 1) * batchSize;
                    NDArray batch =
                            data.get(expression).reshape(new Shape(batchSize, 28 * 28)).div(255f);
                    NDArray labelBatch = label.get(expression);
                    NDArray loss;
                    NDArray pred;
                    try (MxAutograd autograd = new MxAutograd()) {
                        pred = mlp.forward(new NDList(batch)).head();
                        loss =
                                Loss.softmaxCrossEntropyLoss(
                                        pred, labelBatch, 1.f, 0, -1, true, false);
                        autograd.backward((MxNDArray) loss);
                    }
                    Collection<Parameter> params = mlp.getParameters().values();
                    for (Parameter param : params) {
                        NDArray weight = param.getArray();
                        NDArray state = optimizer.createState(0, weight).head();
                        NDArray grad = weight.getGradient();
                        optimizer.update(0, weight, grad, new NDList(state));
                    }
                    acc.update(labelBatch, pred);
                    if (i == numBatches - 1) {
                        lossString = String.valueOf(loss.toFloatArray()[0]);
                    }
                }
                System.out.println("Epoch = " + (epoch + 1) + "  loss = " + lossString); // NOPMD
                System.out.println( // NOPMD
                        "Epoch = "
                                + (epoch + 1)
                                + "  acc = "
                                + acc.getMetric().getValue().toString()); // NOPMD
            }
        }
    }
}
