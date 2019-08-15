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
import org.apache.mxnet.dataset.Mnist;
import org.apache.mxnet.engine.MxAutograd;
import org.apache.mxnet.engine.MxNDArray;
import org.apache.mxnet.engine.lrscheduler.MxLearningRateTracker;
import org.apache.mxnet.engine.optimizer.MxOptimizer;
import org.apache.mxnet.engine.optimizer.Sgd;
import org.apache.mxnet.jna.JnaUtils;
import software.amazon.ai.Block;
import software.amazon.ai.Parameter;
import software.amazon.ai.integration.exceptions.FailedTestException;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.ndarray.NDManager;
import software.amazon.ai.training.Loss;
import software.amazon.ai.training.dataset.Dataset;
import software.amazon.ai.training.metrics.Accuracy;
import software.amazon.ai.training.metrics.LossMetric;
import software.amazon.ai.util.Pair;

public final class MnistUtils {

    private MnistUtils() {}

    public static void trainMnist(
            Block mlp, NDManager manager, int numEpoch, float expectedLoss, float expectedAccuracy)
            throws FailedTestException, IOException {
        // TODO remove numpy flag
        JnaUtils.setNumpyMode(true);

        int batchSize = 100;

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

        Mnist mnist =
                new Mnist.Builder(manager)
                        .setUsage(Dataset.Usage.TRAIN)
                        .setDataLoadingProperty(true, batchSize, false)
                        .build();

        for (int epoch = 0; epoch < numEpoch; epoch++) {
            // reset loss and accuracy
            acc.reset();
            lossMetric.reset();
            NDArray loss;
            for (Pair<NDList, NDList> batch : mnist.getData()) {
                NDArray data = batch.getKey().head().reshape(batchSize, 28 * 28).div(255f);
                NDArray label = batch.getValue().head();
                NDArray pred;
                try (MxAutograd autograd = new MxAutograd()) {
                    pred = mlp.forward(new NDList(data)).head();
                    loss = Loss.softmaxCrossEntropyLoss(label, pred, 1.f, 0, -1, true, false);
                    autograd.backward((MxNDArray) loss);
                }
                Collection<Parameter> params = mlp.getParameters().values();
                for (Parameter param : params) {
                    NDArray weight = param.getArray();
                    NDArray state = optimizer.createState(0, weight).head();
                    NDArray grad = weight.getGradient();
                    optimizer.update(0, weight, grad, new NDList(state));
                }
                acc.update(label, pred);
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
        // TODO remove numpy flag
        JnaUtils.setNumpyMode(false);
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
}
