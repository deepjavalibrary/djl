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
import org.apache.mxnet.dataset.Mnist;
import org.apache.mxnet.jna.JnaUtils;
import software.amazon.ai.Block;
import software.amazon.ai.integration.exceptions.FailedTestException;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.ndarray.NDManager;
import software.amazon.ai.training.Gradient;
import software.amazon.ai.training.Loss;
import software.amazon.ai.training.dataset.Dataset;
import software.amazon.ai.training.dataset.Record;
import software.amazon.ai.training.metrics.Accuracy;
import software.amazon.ai.training.metrics.LossMetric;
import software.amazon.ai.training.optimizer.Optimizer;
import software.amazon.ai.training.optimizer.Sgd;
import software.amazon.ai.training.optimizer.lrscheduler.LrScheduler;

public final class MnistUtils {

    private MnistUtils() {}

    public static void trainMnist(
            Block mlp, NDManager manager, int numEpoch, float expectedLoss, float expectedAccuracy)
            throws FailedTestException, IOException {
        // TODO remove numpy flag
        JnaUtils.setNumpyMode(true);

        int batchSize = 100;

        Optimizer optimizer =
                new Sgd.Builder(mlp.getParameters())
                        .setRescaleGrad(1.0f / batchSize)
                        .setLrScheduler(LrScheduler.fixedLR(0.1f))
                        .setMomentum(0.9f)
                        .build();
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
            for (Record record : mnist.getRecords()) {
                NDArray data = record.getData().head().reshape(batchSize, 28 * 28).div(255f);
                NDArray label = record.getLabels().head();
                NDArray pred;
                try (Gradient.Collector gradCol = Gradient.newCollector()) {
                    Gradient.OptimizerKey optKey = gradCol.collectFor(optimizer);
                    pred = mlp.forward(new NDList(data)).head();
                    loss = Loss.softmaxCrossEntropyLoss(label, pred, 1.f, 0, -1, true, false);
                    optimizer.step(gradCol.collect(loss).get(optKey));
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
