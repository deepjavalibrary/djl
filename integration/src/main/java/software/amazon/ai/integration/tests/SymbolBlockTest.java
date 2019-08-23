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
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.apache.mxnet.engine.SymbolBlock;
import software.amazon.ai.Block;
import software.amazon.ai.Model;
import software.amazon.ai.SequentialBlock;
import software.amazon.ai.integration.IntegrationTest;
import software.amazon.ai.integration.exceptions.FailedTestException;
import software.amazon.ai.integration.util.Assertions;
import software.amazon.ai.integration.util.FileUtils;
import software.amazon.ai.integration.util.RunAsTest;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDArrays;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.ndarray.NDManager;
import software.amazon.ai.ndarray.types.Shape;
import software.amazon.ai.nn.core.Linear;
import software.amazon.ai.training.Gradient;
import software.amazon.ai.training.Loss;
import software.amazon.ai.training.initializer.Initializer;
import software.amazon.ai.util.Pair;

public class SymbolBlockTest {
    public static void main(String[] args) {
        String[] cmd = {"-c", SymbolBlockTest.class.getName()};
        new IntegrationTest()
                .runTests(
                        Stream.concat(Arrays.stream(cmd), Arrays.stream(args))
                                .toArray(String[]::new));
    }

    @RunAsTest
    public void testInference() throws FailedTestException, IOException {
        Path modelPathPrefix = Paths.get(prepareModel() + "/mnist");
        Model mnistmlp = Model.loadModel(modelPathPrefix);
        Block block = mnistmlp.getBlock();
        Shape shape;
        try (NDManager manager = NDManager.newBaseManager().newSubManager()) {
            NDArray arr = manager.ones(new Shape(1, 28, 28));
            shape = block.forward(new NDList(arr)).head().getShape();
        }
        mnistmlp.close();
        Assertions.assertTrue(shape.equals(new Shape(1, 10)));
    }

    @RunAsTest
    public void trainWithNewParam() throws FailedTestException, IOException {
        try (NDManager manager = NDManager.newBaseManager()) {
            Path modelPathPrefix = Paths.get(prepareModel() + "/mnist");
            Model mnistmlp = Model.loadModel(modelPathPrefix);
            Block mlp = mnistmlp.getBlock();
            mlp.setInitializer(manager, Initializer.ONES, true);
            Pair<NDArray, NDArray> result = train(manager, mlp);
            Assertions.assertAlmostEquals(manager.create(6430785.5), result.getKey());
            Assertions.assertAlmostEquals(
                    manager.create(
                            new float[] {
                                2.38418579e-06f,
                                2.38418579e-06f,
                                2.92435288e-05f,
                                3.72529030e-08f,
                                1.43556367e-03f,
                                -2.30967991e-08f
                            }),
                    result.getValue());
            mnistmlp.close();
        }
    }

    @RunAsTest
    public void trainWithExistParam() throws FailedTestException, IOException {
        try (NDManager manager = NDManager.newBaseManager()) {
            Path modelPathPrefix = Paths.get(prepareModel() + "/mnist");
            Model mnistmlp = Model.loadModel(modelPathPrefix);
            Block mlp = mnistmlp.getBlock();
            Pair<NDArray, NDArray> result = train(manager, mlp);
            Assertions.assertAlmostEquals(manager.create(0.29814255237579346), result.getKey());
            Assertions.assertAlmostEquals(
                    manager.create(
                            new float[] {
                                1.51564837e-01f,
                                1.51564837e-01f,
                                9.12832543e-02f,
                                4.07614917e-01f,
                                -1.78348269e-08f,
                                -1.19209291e-08f
                            }),
                    result.getValue());
            mnistmlp.close();
        }
    }

    @RunAsTest
    public void trainWithCustomLayer() throws FailedTestException, IOException {
        try (NDManager manager = NDManager.newBaseManager()) {
            Path modelPathPrefix = Paths.get(prepareModel() + "/mnist");
            Model mnistmlp = Model.loadModel(modelPathPrefix);
            SymbolBlock mlp = (SymbolBlock) mnistmlp.getBlock();
            SymbolBlock mlpPartial = mlp.getLayer("hybridsequential0_dense1_relu_fwd_output");
            SequentialBlock newMlp = new SequentialBlock();
            newMlp.add(mlpPartial);
            newMlp.add(
                    new Linear.Builder()
                            .setOutChannels(10)
                            .build()
                            .setInitializer(manager, Initializer.ONES, true));
            Pair<NDArray, NDArray> result = train(manager, mlp);
            Assertions.assertAlmostEquals(manager.create(0.29814255237579346), result.getKey());
            Assertions.assertAlmostEquals(
                    manager.create(
                            new float[] {
                                1.51564837e-01f,
                                1.51564837e-01f,
                                9.12832543e-02f,
                                4.07614917e-01f,
                                -1.78348269e-08f,
                                -1.19209291e-08f
                            }),
                    result.getValue());
            mnistmlp.close();
        }
    }

    private Pair<NDArray, NDArray> train(NDManager manager, Block mlp) {
        NDArray data = manager.ones(new Shape(10, 28 * 28));
        NDArray label = manager.arange(0, 10);
        NDArray gradMean;
        NDArray pred;
        try (Gradient.Collector gradCol = Gradient.newCollector()) {
            pred = mlp.forward(new NDList(data)).head();
            NDArray loss = Loss.softmaxCrossEntropyLoss(label, pred, 1.f, 0, -1, true, false);
            gradCol.backward(loss);
        }
        List<NDArray> grads =
                mlp.getParameters()
                        .stream()
                        .map(
                                stringParameterPair ->
                                        stringParameterPair.getValue().getArray().getGradient())
                        .collect(Collectors.toCollection(ArrayList::new));
        gradMean = NDArrays.stack(grads.stream().map(NDArray::mean).toArray(NDArray[]::new));
        return new Pair<>(pred.mean(), gradMean);
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
