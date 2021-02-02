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
package ai.djl.training;

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.training.dataset.Batch;
import ai.djl.training.listener.TrainingListener.BatchData;
import java.util.ArrayList;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/** Helper for easy training of a whole model on multiple GPUs in parallel. */
@SuppressWarnings("PMD")
public final class ParallelTrain {

    private final ExecutorService executor;

    /**
     * Build a ParallelTrain for the given devices.
     *
     * @param devices the devices to train on.
     */
    public ParallelTrain(Device[] devices) {
        this.executor = Executors.newFixedThreadPool(devices.length);
    }

    /**
     * Trains the model with one iteration of the given {@link Batch} of data.
     *
     * @param trainer the trainer to validate the batch with
     * @param batch a {@link Batch} that contains data, and its respective labels
     * @throws IllegalArgumentException if the batch engine does not match the trainer engine
     */
    public void trainBatch(Trainer trainer, Batch batch) {
        if (trainer.getManager().getEngine() != batch.getManager().getEngine()) {
            throw new IllegalArgumentException(
                    "The data must be on the same engine as the trainer. You may need to change one of your NDManagers.");
        }
        Batch[] splits = batch.split(trainer.getDevices(), false);
        BatchData batchData =
                new BatchData(batch, new ConcurrentHashMap<>(), new ConcurrentHashMap<>());
        ArrayList<Future<Boolean>> futures = new ArrayList<>(splits.length);
        for (Batch split : splits) {
            futures.add(
                    executor.submit(
                            () -> {
                                try (GradientCollector collector = trainer.newGradientCollector()) {
                                    NDList data = trainer.getDataManager().getData(split);
                                    NDList labels = trainer.getDataManager().getLabels(split);
                                    NDList preds = trainer.forward(data);
                                    long time = System.nanoTime();
                                    NDArray lossValue = trainer.getLoss().evaluate(labels, preds);
                                    collector.backward(lossValue);
                                    trainer.addMetric("backward", time);
                                    time = System.nanoTime();
                                    batchData.getLabels().put(labels.get(0).getDevice(), labels);
                                    batchData.getPredictions().put(preds.get(0).getDevice(), preds);
                                    trainer.addMetric("training-metrics", time);
                                    return true;
                                }
                            }));
        }
        for (Future<Boolean> future : futures) {
            try {
                future.get();
            } catch (InterruptedException e) {
                e.printStackTrace();
            } catch (ExecutionException e) {
                e.printStackTrace();
            }
        }
        trainer.notifyListeners(listener -> listener.onTrainingBatch(trainer, batchData));
    }
}
