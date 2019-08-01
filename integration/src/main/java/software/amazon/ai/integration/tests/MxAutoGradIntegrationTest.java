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

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.Collection;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;
import org.apache.mxnet.engine.MxAutograd;
import org.apache.mxnet.engine.MxNDArray;
import org.apache.mxnet.engine.lrscheduler.MxLearningRateTracker;
import org.apache.mxnet.engine.optimizer.MxOptimizer;
import org.apache.mxnet.engine.optimizer.Sgd;
import software.amazon.ai.Parameter;
import software.amazon.ai.SequentialBlock;
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

    @RunAsTest
    public void testTrain() throws FailedTestException {
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
            float lossValue = loss.getFloat();
            float expectedLoss = 0.001f;
            Assertions.assertTrue(
                    lossValue < expectedLoss,
                    String.format(
                            "Loss did not improve, loss value: %f, expected "
                                    + "max loss value: %f",
                            lossValue, expectedLoss));
        }
    }

    // @RunAsTest
    public void testTrainMnist() throws IOException, FailedTestException {

        try (NDManager manager = NDManager.newBaseManager()) {

            // TODO: Remove loading mnist with DataLoader
            String source = "http://data.mxnet.io/mxnet/data/mnist.zip";
            String dataDir = System.getProperty("user.home") + "/.joule_data";
            String downloadDestination = dataDir + "/mnist.zip";
            String extractDestination = dataDir + "/mnist";
            Path trainImages = Paths.get(extractDestination + "/train-images-idx3-ubyte");
            Path trainLabels = Paths.get(extractDestination + "/train-labels-idx1-ubyte");
            // download and unzip data if not exist
            if (!Files.exists(trainImages) || !Files.exists(trainLabels)) {
                if (!Files.exists(Paths.get(downloadDestination))) {
                    download(source, dataDir, "mnist.zip");
                }
                unzip(downloadDestination, extractDestination);
                deleteFileOrDir(downloadDestination);
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

            SequentialBlock mlp = new SequentialBlock();
            mlp.add(new Linear.Builder().setOutChannels(128).build());
            mlp.add(Activation.reluBlock());
            mlp.add(new Linear.Builder().setOutChannels(64).build());
            mlp.add(Activation.reluBlock());
            mlp.add(new Linear.Builder().setOutChannels(10).build());
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

            float lossSum = 0.f;
            Accuracy acc = new Accuracy();
            for (int epoch = 0; epoch < numEpoch; epoch++) {
                // reset loss and accuracy
                acc.reset();
                lossSum = 0.f;
                NDArray loss;
                for (int i = 0; i < numBatches; i++) {
                    String expression = i * batchSize + ":" + (i + 1) * batchSize;
                    NDArray batch =
                            data.get(expression).reshape(new Shape(batchSize, 28 * 28)).div(255f);
                    NDArray labelBatch = label.get(expression);
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
                    // sum all loss for the batch
                    lossSum += loss.sum().getFloat();
                }
            }
            // final loss is sum of all loss divided by num of data
            float lossValue = lossSum / data.getShape().get(0);
            float accuracy = acc.getMetric().getValue().floatValue();
            float expectedLoss = 0.1f;
            float expectedAccuracy = 0.9f;
            if (lossValue > expectedLoss) {
                throw new FailedTestException(
                        String.format(
                                "Loss did not improve, loss value: %f, expected "
                                        + "maximal loss value: %f",
                                lossValue, expectedLoss));
            }
            if (accuracy < expectedAccuracy) {
                throw new FailedTestException(
                        String.format(
                                "Accuracy did not improve, accuracy value: %f, expected "
                                        + "minimal accuracy value: %f",
                                accuracy, expectedAccuracy));
            }
        }
    }

    public void download(String source, String destination, String fileName) throws IOException {
        URL url = new URL(source);
        InputStream in = url.openStream();
        File destDir = new File(destination);
        if (!destDir.exists()) {
            if (!destDir.mkdir()) {
                throw new IOException("Failed to create directory: " + destDir);
            }
        }
        Files.copy(
                in, Paths.get(destination + "/" + fileName), StandardCopyOption.REPLACE_EXISTING);
    }

    public void unzip(String zipFilePath, String destDirectory) throws IOException {
        File destDir = new File(destDirectory);
        if (!destDir.exists()) {
            if (!destDir.mkdir()) {
                throw new IOException("Failed to create directory: " + destDirectory);
            }
        }
        ZipInputStream zipIn = new ZipInputStream(Files.newInputStream(Paths.get(zipFilePath)));
        ZipEntry entry = zipIn.getNextEntry();
        // iterates over entries in the zip file
        while (entry != null) {
            String filePath = destDirectory + File.separator + entry.getName();
            if (!entry.isDirectory()) {
                // if the entry is a file, extracts it
                extractFile(zipIn, filePath);
            } else {
                // if the entry is a directory, make the directory
                File dir = new File(filePath);
                if (!dir.mkdir()) {
                    throw new IOException("Failed to create directory: " + filePath);
                }
            }
            zipIn.closeEntry();
            entry = zipIn.getNextEntry();
        }
        zipIn.close();
    }

    private void extractFile(ZipInputStream zipIn, String filePath) throws IOException {
        OutputStream out = Files.newOutputStream(Paths.get(filePath));
        byte[] bytesIn = new byte[4096];
        int read;
        while ((read = zipIn.read(bytesIn)) != -1) {
            out.write(bytesIn, 0, read);
        }
        out.close();
    }

    public void deleteFileOrDir(String target) throws IOException {
        File dir = new File(target);
        if (Files.isDirectory(dir.toPath())) {
            String[] entries = dir.list();
            if (entries != null) {
                for (String s : entries) {
                    File currentFile = new File(dir.getPath(), s);
                    if (!currentFile.delete()) {
                        throw new IOException("Failed to delete " + s);
                    }
                }
            }
        }
        if (!dir.delete()) {
            throw new IOException("Failed to delete " + target);
        }
    }
}
