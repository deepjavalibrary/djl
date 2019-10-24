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
package ai.djl.integration.tests.nn;

import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.integration.util.Assertions;
import ai.djl.mxnet.engine.MxGradientCollector;
import ai.djl.mxnet.zoo.MxModelZoo;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.SymbolBlock;
import ai.djl.nn.core.Linear;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.GradientCollector;
import ai.djl.training.ParameterStore;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.training.initializer.Initializer;
import ai.djl.training.loss.Loss;
import ai.djl.util.Pair;
import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;
import org.testng.Assert;
import org.testng.annotations.Test;

public class SymbolBlockTest {

    @Test
    public void testForward() throws IOException, ModelNotFoundException, MalformedModelException {
        Map<String, String> criteria = new ConcurrentHashMap<>();
        try (Model model = MxModelZoo.MLP.loadModel(criteria)) {
            NDManager manager = model.getNDManager();

            ParameterStore parameterStore = new ParameterStore(manager, false);

            Block block = model.getBlock();
            NDArray arr = manager.ones(new Shape(1, 28, 28));
            Shape shape =
                    block.forward(parameterStore, new NDList(arr)).singletonOrThrow().getShape();
            Assert.assertEquals(new Shape(1, 10), shape);
        }
    }

    @Test
    public void trainWithNewParam()
            throws IOException, ModelNotFoundException, MalformedModelException {
        TrainingConfig config =
                new DefaultTrainingConfig(Initializer.ONES, Loss.softmaxCrossEntropyLoss());
        Map<String, String> criteria = new ConcurrentHashMap<>();
        try (Model model = MxModelZoo.MLP.loadModel(criteria)) {

            model.getBlock().clear();
            try (Trainer trainer = model.newTrainer(config)) {
                NDManager manager = trainer.getManager();

                Pair<NDArray, NDArray> result = train(manager, trainer, model.getBlock());
                Assertions.assertAlmostEquals(result.getKey(), manager.create(6422528.0));
                Assertions.assertAlmostEquals(
                        result.getValue(),
                        manager.create(
                                new float[] {
                                    2.38418579e-06f,
                                    2.38418579e-06f,
                                    2.92062759e-05f,
                                    3.72529030e-08f,
                                    -4.03289776e-03f,
                                    -2.30967991e-08f
                                }));
            }
        }
    }

    @Test
    public void trainWithExistParam()
            throws IOException, ModelNotFoundException, MalformedModelException {
        TrainingConfig config =
                new DefaultTrainingConfig(Initializer.ONES, Loss.softmaxCrossEntropyLoss());
        Map<String, String> criteria = new ConcurrentHashMap<>();
        try (Model model = MxModelZoo.MLP.loadModel(criteria)) {

            try (Trainer trainer = model.newTrainer(config)) {
                NDManager manager = trainer.getManager();

                Pair<NDArray, NDArray> result = train(manager, trainer, model.getBlock());
                Assertions.assertAlmostEquals(result.getKey(), manager.create(0.29814255237579346));
                Assertions.assertAlmostEquals(
                        result.getValue(),
                        manager.create(
                                new float[] {
                                    1.51564837e-01f,
                                    1.51564837e-01f,
                                    9.12832543e-02f,
                                    4.07614917e-01f,
                                    -1.78348269e-08f,
                                    -1.19209291e-08f
                                }));
            }
        }
    }

    @Test
    public void trainWithCustomLayer()
            throws IOException, ModelNotFoundException, MalformedModelException {
        TrainingConfig config =
                new DefaultTrainingConfig(Initializer.ONES, Loss.softmaxCrossEntropyLoss());
        Map<String, String> criteria = new ConcurrentHashMap<>();
        try (Model model = MxModelZoo.MLP.loadModel(criteria)) {
            NDManager manager = model.getNDManager();

            SymbolBlock mlp = (SymbolBlock) model.getBlock();
            SequentialBlock newMlp = new SequentialBlock();
            mlp.removeLastBlock();
            newMlp.add(mlp);
            Linear linear = new Linear.Builder().setOutChannels(10).build();

            linear.setInitializer(Initializer.ONES);
            newMlp.add(linear);

            model.setBlock(newMlp);

            try (Trainer trainer = model.newTrainer(config)) {
                Pair<NDArray, NDArray> result = train(manager, trainer, newMlp);
                Assertions.assertAlmostEquals(result.getKey(), manager.create(17.357540130615234));
                Assertions.assertAlmostEquals(
                        result.getValue(),
                        manager.create(
                                new float[] {
                                    1.54082624e-09f,
                                    1.54082624e-09f,
                                    3.12847304e-09f,
                                    1.39698386e-08f,
                                    -7.56020135e-09f,
                                    -2.30967991e-08f
                                }));
            }
        }
    }

    private Pair<NDArray, NDArray> train(NDManager manager, Trainer trainer, Block block) {
        Shape inputShape = new Shape(10, 28 * 28);
        trainer.initialize(inputShape);

        NDArray data = manager.ones(inputShape);
        NDArray label = manager.arange(0, 10);
        NDArray pred;
        try (GradientCollector gradCol = new MxGradientCollector()) {
            pred = trainer.forward(new NDList(data)).singletonOrThrow();
            NDArray loss =
                    Loss.softmaxCrossEntropyLoss().getLoss(new NDList(label), new NDList(pred));
            gradCol.backward(loss);
        }
        List<NDArray> grads =
                block.getParameters()
                        .stream()
                        .map(
                                stringParameterPair ->
                                        stringParameterPair.getValue().getArray().getGradient())
                        .collect(Collectors.toList());
        NDArray gradMean =
                NDArrays.stack(
                        new NDList(grads.stream().map(NDArray::mean).toArray(NDArray[]::new)));
        return new Pair<>(pred.mean(), gradMean);
    }
}
