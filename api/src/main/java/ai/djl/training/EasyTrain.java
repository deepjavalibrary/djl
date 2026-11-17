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

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.training.dataset.Batch;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.listener.TrainingListener.BatchData;
import ai.djl.translate.TranslateException;
import ai.djl.util.Preconditions;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.function.Consumer;

/** Helper for easy training of a whole model, a trainining batch, or a validation batch. */
public final class EasyTrain {

    private EasyTrain() {}

    /**
     * Runs a basic epoch training experience with a given trainer.
     *
     * @param trainer the trainer to train for
     * @param numEpoch the number of epochs to train
     * @param trainingDataset the dataset to train on
     * @param validateDataset the dataset to validate against. Can be null for no validation
     * @throws IOException for various exceptions depending on the dataset
     * @throws TranslateException if there is an error while processing input
     */
    public static void fit(
            Trainer trainer, int numEpoch, Dataset trainingDataset, Dataset validateDataset)
            throws IOException, TranslateException {

        // Deep learning is typically trained in epochs where each epoch trains the model on each
        // item in the dataset once
        for (int epoch = 0; epoch < numEpoch; epoch++) {

            // We iterate through the dataset once during each epoch
            for (Batch batch : trainer.iterateDataset(trainingDataset)) {

                try {
                    trainBatch(trainer, batch);
                    trainer.step();
                } finally {
                    batch.close();
                }
            }

            // After each epoch, test against the validation dataset if we have one
            evaluateDataset(trainer, validateDataset);

            // reset training and validation evaluators at end of epoch
            trainer.notifyListeners(listener -> listener.onEpoch(trainer));
        }
    }

    /**
     * Trains the model with one iteration of the given {@link Batch} of data.
     *
     * @param trainer the trainer to validate the batch with
     * @param batch a {@link Batch} that contains data, and its respective labels
     * @throws IllegalArgumentException if the batch engine does not match the trainer engine
     */
    public static void trainBatch(Trainer trainer, Batch batch) {
        if (trainer.getManager().getEngine() != batch.getManager().getEngine()) {
            throw new IllegalArgumentException(
                    "The data must be on the same engine as the trainer. You may need to change one"
                            + " of your NDManagers.");
        }
        Batch[] splits = batch.split(trainer.getDevices(), false);
        BatchData batchData =
                new BatchData(batch, new ConcurrentHashMap<>(), new ConcurrentHashMap<>());
        if (splits.length > 1 && trainer.getExecutorService().isPresent()) {
            runParallel(
                    splits,
                    trainer.getExecutorService().get(),
                    split -> {
                        // Gradient recording belongs to the thread running the forward pass.
                        try (GradientCollector collector = trainer.newGradientCollector()) {
                            trainSplit(trainer, collector, batchData, split);
                        }
                    });
        } else {
            try (GradientCollector collector = trainer.newGradientCollector()) {
                for (Batch split : splits) {
                    trainSplit(trainer, collector, batchData, split);
                }
            }
        }

        trainer.notifyListeners(listener -> listener.onTrainingBatch(trainer, batchData));
    }

    private static void trainSplit(
            Trainer trainer, GradientCollector collector, BatchData batchData, Batch split) {
        NDList data = split.getData();
        NDList labels = split.getLabels();
        NDList preds = trainer.forward(data, labels);
        long time = System.nanoTime();
        NDArray lossValue = trainer.getLoss().evaluate(labels, preds);
        collector.backward(lossValue);
        trainer.addMetric("backward", time);
        time = System.nanoTime();
        batchData.getLabels().put(labels.get(0).getDevice(), labels);
        batchData.getPredictions().put(preds.get(0).getDevice(), preds);
        trainer.addMetric("training-metrics", time);
    }

    /**
     * Validates the given batch of data.
     *
     * <p>During validation, the evaluators and losses are computed, but gradients aren't computed,
     * and parameters aren't updated.
     *
     * @param trainer the trainer to validate the batch with
     * @param batch a {@link Batch} of data
     * @throws IllegalArgumentException if the batch engine does not match the trainer engine
     */
    public static void validateBatch(Trainer trainer, Batch batch) {
        Preconditions.checkArgument(
                trainer.getManager().getEngine() == batch.getManager().getEngine(),
                "The data must be on the same engine as the trainer. You may need to change one of"
                        + " your NDManagers.");
        Batch[] splits = batch.split(trainer.getDevices(), false);
        BatchData batchData =
                new BatchData(batch, new ConcurrentHashMap<>(), new ConcurrentHashMap<>());

        if (splits.length > 1 && trainer.getExecutorService().isPresent()) {
            runParallel(
                    splits,
                    trainer.getExecutorService().get(),
                    split -> validateSplit(trainer, batchData, split));
        } else {
            // sequence
            for (Batch split : splits) {
                validateSplit(trainer, batchData, split);
            }
        }

        trainer.notifyListeners(listener -> listener.onValidationBatch(trainer, batchData));
    }

    private static void validateSplit(Trainer trainer, BatchData batchData, Batch split) {
        NDList data = split.getData();
        NDList labels = split.getLabels();
        NDList preds = trainer.evaluate(data);
        batchData.getLabels().put(labels.get(0).getDevice(), labels);
        batchData.getPredictions().put(preds.get(0).getDevice(), preds);
    }

    private static void runParallel(
            Batch[] splits, ExecutorService executor, Consumer<Batch> action) {
        List<CompletableFuture<Void>> futures = new ArrayList<>(splits.length);
        try {
            for (Batch split : splits) {
                futures.add(CompletableFuture.runAsync(() -> action.accept(split), executor));
            }
        } finally {
            // Even if submission fails, let accepted tasks finish using the batch before returning.
            CompletableFuture.allOf(futures.toArray(new CompletableFuture<?>[0])).join();
        }
    }

    /**
     * Evaluates the test dataset.
     *
     * @param trainer the trainer to evaluate on
     * @param testDataset the test dataset to evaluate
     * @throws IOException for various exceptions depending on the dataset
     * @throws TranslateException if there is an error while processing input
     */
    public static void evaluateDataset(Trainer trainer, Dataset testDataset)
            throws IOException, TranslateException {

        if (testDataset != null) {
            for (Batch batch : trainer.iterateDataset(testDataset)) {
                try {
                    validateBatch(trainer, batch);
                } finally {
                    batch.close();
                }
            }
        }
    }
}
