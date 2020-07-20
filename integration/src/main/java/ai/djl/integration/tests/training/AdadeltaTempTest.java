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
package ai.djl.integration.tests.training;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.basicdataset.AirfoilRandomAccess;
import ai.djl.metric.Metrics;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.testing.Assertions;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.training.dataset.Batch;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.initializer.Initializer;
import ai.djl.training.initializer.NormalInitializer;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.translate.Batchifier;
import ai.djl.translate.TranslateException;
import java.io.IOException;
import org.testng.annotations.Test;

@SuppressWarnings("MissingJavadocMethod")
public class AdadeltaTempTest {

    private static final int BATCH_SIZE = 10;
    private static final int CHANNELS = 10;

    @Test
    public void testAdadelta() {
        Optimizer optim = Optimizer.adadelta().build();

        Device[] devices = Device.getDevices(1);
        TrainingConfig config =
                new DefaultTrainingConfig(Loss.l2Loss())
                        .optInitializer(Initializer.ONES)
                        .optOptimizer(optim)
                        .optDevices(devices);
        Block block = Linear.builder().setUnits(CHANNELS).build();
        try (Model model = Model.newInstance("model", devices[0])) {
            model.setBlock(block);

            try (Trainer trainer = model.newTrainer(config)) {
                int batchSize = config.getDevices().length * BATCH_SIZE;
                trainer.initialize(new Shape(batchSize, CHANNELS));

                NDManager manager = trainer.getManager();
                NDArray result = runOptimizer(manager, trainer, block, batchSize);
                NDArray result2 = runOptimizer(manager, trainer, block, batchSize);

                //                System.out.println(result);
                //                System.out.println(result2);
                Assertions.assertAlmostEquals(result, manager.create(new float[] {0.999f, 0f}));
                Assertions.assertAlmostEquals(result2, manager.create(new float[] {0.999f, 0f}));
            }
        }
    }

    public static AirfoilRandomAccess getDataCh11(int batchSize, int n)
            throws IOException, TranslateException {
        // Load data
        AirfoilRandomAccess airfoil =
                AirfoilRandomAccess.builder()
                        .optUsage(Dataset.Usage.TRAIN)
                        .setSampling(batchSize, true)
                        .build();
        // Select Features
        airfoil.addAllFeatures();
        // Prepare Data
        airfoil.prepare();
        // Select first n cases
        airfoil.selectFirstN(n);
        // Remove the mean and rescale variance to 1 for all features
        airfoil.whitenAll();
        return airfoil;
    }

    private NDArray runOptimizer(NDManager manager, Trainer trainer, Block block, int batchSize) {
        NDArray data = manager.ones(new Shape(batchSize, CHANNELS)).mul(2);
        NDArray label = data.mul(2);
        Batch batch =
                new Batch(
                        manager,
                        new NDList(data),
                        new NDList(label),
                        batchSize,
                        Batchifier.STACK,
                        Batchifier.STACK);
        EasyTrain.trainBatch(trainer, batch);
        trainer.step();
        return NDArrays.stack(
                new NDList(
                        block.getParameters()
                                .stream()
                                .map(paramPair -> paramPair.getValue().getArray().mean())
                                .toArray(NDArray[]::new)));
    }

    public static void trainConciseCh11(Optimizer sgd, AirfoilRandomAccess dataset, int numEpochs)
            throws IOException, TranslateException {
        // Initialization
        SequentialBlock net = new SequentialBlock();
        Linear linear = Linear.builder().setUnits(1).build();
        net.add(linear);
        net.setInitializer(new NormalInitializer());

        Model model = Model.newInstance("concise implementation");
        model.setBlock(net);

        Loss loss = Loss.l2Loss();

        DefaultTrainingConfig config =
                new DefaultTrainingConfig(loss)
                        .optOptimizer(sgd)
                        .addEvaluator(new Accuracy()) // Model Accuracy
                        .addTrainingListeners(TrainingListener.Defaults.logging()); // Logging

        Trainer trainer = model.newTrainer(config);

        trainer.initialize(new Shape(10, 5));

        Metrics metrics = new Metrics();
        trainer.setMetrics(metrics);

        for (int i = 0; i < numEpochs; i++) {
            for (Batch batch : trainer.iterateDataset(dataset)) {
                EasyTrain.trainBatch(trainer, batch);
                trainer.step();

                batch.close();
            }
        }
    }
    /* End Ch11 Optimization */

    @Test
    public void testAdadelta2() throws IOException, TranslateException {
        Optimizer optim = Optimizer.adadelta().build();

        AirfoilRandomAccess airfoil = getDataCh11(32, 1500);
        trainConciseCh11(optim, airfoil, 20);
    }
}
