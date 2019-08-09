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

package software.amazon.ai.integration.util;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Collection;
import org.apache.mxnet.engine.MxAutograd;
import org.apache.mxnet.engine.MxNDArray;
import org.apache.mxnet.engine.lrscheduler.MxLearningRateTracker;
import org.apache.mxnet.engine.optimizer.MxOptimizer;
import org.apache.mxnet.engine.optimizer.Sgd;
import software.amazon.ai.Block;
import software.amazon.ai.Parameter;
import software.amazon.ai.integration.exceptions.FailedTestException;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.ndarray.NDManager;
import software.amazon.ai.ndarray.types.DataType;
import software.amazon.ai.ndarray.types.Shape;
import software.amazon.ai.training.Loss;
import software.amazon.ai.training.metrics.Accuracy;
import software.amazon.ai.training.metrics.LossMetric;
import software.amazon.ai.util.Pair;

public final class MnistUtils {

    private MnistUtils() {}

    public static void trainMnist(
            Block mlp, NDManager manager, int numEpoch, float expectedLoss, float expectedAccuracy)
            throws FailedTestException, IOException {
        // TODO: Remove loading mnist with DataLoader
        Pair<NDArray, NDArray> dataLabel = mnistSetup(manager);
        NDArray data = dataLabel.getKey();
        NDArray label = dataLabel.getValue();

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

        Accuracy acc = new Accuracy();
        LossMetric lossMetric = new LossMetric("softmaxCELoss");

        for (int epoch = 0; epoch < numEpoch; epoch++) {
            // reset loss and accuracy
            acc.reset();
            lossMetric.reset();
            NDArray loss;
            for (int i = 0; i < numBatches; i++) {
                String expression = i * batchSize + ":" + (i + 1) * batchSize;
                NDArray batch =
                        data.get(expression).reshape(new Shape(batchSize, 28 * 28)).div(255f);
                NDArray labelBatch = label.get(expression);
                NDArray pred;
                try (MxAutograd autograd = new MxAutograd()) {
                    pred = mlp.forward(new NDList(batch)).head();
                    loss = Loss.softmaxCrossEntropyLoss(labelBatch, pred, 1.f, 0, -1, true, false);
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
                lossMetric.update(loss);
            }
        }
        // final loss is sum of all loss divided by num of data
        float lossValue = lossMetric.getMetric().getValue();
        float accuracy = acc.getMetric().getValue();
        Assertions.assertTrue(
                lossValue <= expectedLoss,
                String.format(
                        "Loss did not improve, loss value: %f, expected "
                                + "maximal loss value: %f",
                        lossValue, expectedLoss));
        Assertions.assertTrue(
                accuracy >= expectedAccuracy,
                String.format(
                        "Accuracy did not improve, accuracy value: %f, expected "
                                + "minimal accuracy value: %f",
                        accuracy, expectedAccuracy));
    }

    public static String prepareModel() throws IOException {
        String source = "https://joule.s3.amazonaws.com/other+resources/mnistmlp.zip";
        String dataDir = System.getProperty("user.home") + "/.joule_data";
        String downloadDestination = dataDir + "/mnistmlp.zip";
        String extractDestination = dataDir + "/mnist";
        Path params = Paths.get(extractDestination + "/mnist-0000.params");
        Path symbol = Paths.get(extractDestination + "/mnist-symbol.json");
        // download and unzip data if not exist
        if (!Files.exists(params) || !Files.exists(symbol)) {
            if (!Files.exists(Paths.get(downloadDestination))) {
                FileUtils.download(source, dataDir, "mnistmlp.zip");
            }
            FileUtils.unzip(downloadDestination, extractDestination);
            FileUtils.deleteFileOrDir(downloadDestination);
        }
        return extractDestination;
    }

    public static Pair<NDArray, NDArray> mnistSetup(NDManager manager) throws IOException {
        String source = "http://data.mxnet.io/mxnet/data/mnist.zip";
        String dataDir = System.getProperty("user.home") + "/.joule_data";
        String downloadDestination = dataDir + "/mnist.zip";
        String extractDestination = dataDir + "/mnist";
        Path trainImages = Paths.get(extractDestination + "/train-images-idx3-ubyte");
        Path trainLabels = Paths.get(extractDestination + "/train-labels-idx1-ubyte");
        // download and unzip data if not exist
        if (!Files.exists(trainImages) || !Files.exists(trainLabels)) {
            if (!Files.exists(Paths.get(downloadDestination))) {
                FileUtils.download(source, dataDir, "mnist.zip");
            }
            FileUtils.unzip(downloadDestination, extractDestination);
            FileUtils.deleteFileOrDir(downloadDestination);
        }
        byte[] imageBytesRaw = Files.readAllBytes(trainImages);
        byte[] labelBytesRaw = Files.readAllBytes(trainLabels);

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

        return new Pair<>(data, label);
    }
}
