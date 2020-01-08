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
package ai.djl.examples.training.util;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.TrainingDivergedException;
import ai.djl.engine.Engine;
import ai.djl.examples.util.MemoryUtils;
import ai.djl.metric.Metrics;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingListener;
import ai.djl.training.evaluator.Evaluator;
import ai.djl.training.loss.Loss;
import ai.djl.training.util.ProgressBar;
import java.util.ArrayList;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class ExampleTrainingListener implements TrainingListener {

    private static final Logger logger = LoggerFactory.getLogger(ExampleTrainingListener.class);

    private int batchSize;
    private int trainDataSize;
    private int validateDataSize;
    private int trainingProgress;
    private int validateProgress;

    private long epochTime;
    private int numEpochs;
    private String outputDir;

    private ProgressBar trainingProgressBar;
    private ProgressBar validateProgressBar;

    private ExampleTrainingListener(
            int batchSize, int trainDataSize, int validateDataSize, String outputDir) {
        this.batchSize = batchSize;
        this.trainDataSize = trainDataSize;
        this.validateDataSize = validateDataSize;
        this.outputDir = outputDir;
    }

    public static TrainingListener[] exampleListeners(
            int batchSize,
            int trainDataSize,
            int validateDataSize,
            int maxGpus,
            int epoch,
            String outputDir) {
        ExampleTrainingListener exampleTrainingListener =
                new ExampleTrainingListener(batchSize, trainDataSize, validateDataSize, outputDir);
        exampleTrainingListener.beforeTrain(maxGpus, epoch);
        return new TrainingListener[] {exampleTrainingListener};
    }

    /** {@inheritDoc} */
    @Override
    public void onEpoch(Trainer trainer) {
        Metrics metrics = trainer.getMetrics();
        if (epochTime > 0L) {
            metrics.addMetric("epoch", System.nanoTime() - epochTime);
        }
        logger.info("Epoch " + numEpochs + " finished.");
        printTrainingStatus(trainer);

        epochTime = System.nanoTime();
        numEpochs++;
        trainingProgress = 0;
        validateProgress = 0;
    }

    /** {@inheritDoc} */
    @Override
    public void onTrainingBatch(Trainer trainer) {
        Metrics metrics = trainer.getMetrics();
        MemoryUtils.collectMemoryInfo(metrics);

        Loss trainingLoss = trainer.getLoss();
        if (Float.isNaN(trainingLoss.getValue())) {
            throw new TrainingDivergedException(
                    "The Loss became NaN, try reduce learning rate,"
                            + "add clipGradient option to your optimizer, check input data and loss calculation.");
        }

        for (Evaluator evaluator : trainer.getTrainingEvaluators()) {
            metrics.addMetric("train_" + evaluator.getName(), evaluator.getValue());
        }

        if (trainingProgressBar == null) {
            trainingProgressBar = new ProgressBar("Training", trainDataSize);
        }
        trainingProgressBar.update(trainingProgress++, getTrainingStatus(trainer));
    }

    /** {@inheritDoc} */
    @Override
    public void onValidationBatch(Trainer trainer) {
        Metrics metrics = trainer.getMetrics();
        MemoryUtils.collectMemoryInfo(metrics);

        for (Evaluator evaluator : trainer.getValidationEvaluators()) {
            metrics.addMetric("validate_" + evaluator.getName(), evaluator.getValue());
        }

        if (validateProgressBar == null) {
            validateProgressBar = new ProgressBar("Validating", validateDataSize);
        }
        validateProgressBar.update(validateProgress++);
    }

    /** {@inheritDoc} */
    @Override
    public void onTrainingEnd(Trainer trainer) {
        afterTrain(trainer, outputDir);
        recordPerformance(trainer);
    }

    public void beforeTrain(int maxGpus, int epoch) {
        String devices;
        if (maxGpus > 0) {
            devices = maxGpus + " GPUs";
        } else {
            devices = Device.cpu().toString();
        }
        logger.info("Running {} on: {}, epoch: {}.", getClass().getSimpleName(), devices, epoch);

        long init = System.nanoTime();
        String version = Engine.getInstance().getVersion();
        long loaded = System.nanoTime();
        logger.info(
                String.format(
                        "Load library %s in %.3f ms.", version, (loaded - init) / 1_000_000f));
        epochTime = System.nanoTime();
    }

    public void afterTrain(Trainer trainer, String outputDir) {
        Metrics metrics = trainer.getMetrics();
        logger.info("Training: {} batches", trainDataSize);
        logger.info("Validation: {} batches", validateDataSize);

        if (metrics.hasMetric("train")) {
            // possible no train metrics if only one iteration is executed
            float p50 = metrics.percentile("train", 50).getValue().longValue() / 1_000_000f;
            float p90 = metrics.percentile("train", 90).getValue().longValue() / 1_000_000f;
            logger.info(String.format("train P50: %.3f ms, P90: %.3f ms", p50, p90));
        }

        float p50 = metrics.percentile("forward", 50).getValue().longValue() / 1_000_000f;
        float p90 = metrics.percentile("forward", 90).getValue().longValue() / 1_000_000f;
        logger.info(String.format("forward P50: %.3f ms, P90: %.3f ms", p50, p90));

        p50 = metrics.percentile("training-metrics", 50).getValue().longValue() / 1_000_000f;
        p90 = metrics.percentile("training-metrics", 90).getValue().longValue() / 1_000_000f;
        logger.info(String.format("training-metrics P50: %.3f ms, P90: %.3f ms", p50, p90));

        p50 = metrics.percentile("backward", 50).getValue().longValue() / 1_000_000f;
        p90 = metrics.percentile("backward", 90).getValue().longValue() / 1_000_000f;
        logger.info(String.format("backward P50: %.3f ms, P90: %.3f ms", p50, p90));

        p50 = metrics.percentile("step", 50).getValue().longValue() / 1_000_000f;
        p90 = metrics.percentile("step", 90).getValue().longValue() / 1_000_000f;
        logger.info(String.format("step P50: %.3f ms, P90: %.3f ms", p50, p90));

        p50 = metrics.percentile("epoch", 50).getValue().longValue() / 1_000_000_000f;
        p90 = metrics.percentile("epoch", 90).getValue().longValue() / 1_000_000_000f;
        logger.info(String.format("epoch P50: %.3f s, P90: %.3f s", p50, p90));

        if (outputDir != null) {
            MemoryUtils.dumpMemoryInfo(metrics, outputDir);
            TrainingUtils.dumpTrainingTimeInfo(metrics, outputDir);
        }
    }

    public void recordPerformance(Trainer trainer) {
        // TODO: Add more property into model: Throughput, Memory, mAP, Dataset etc.
        Model model = trainer.getModel();
        Metrics metrics = trainer.getMetrics();
        model.setProperty("Epoch", Integer.toString(numEpochs));
        for (Evaluator evaluator : trainer.getValidationEvaluators()) {
            float value =
                    metrics.latestMetric("validate_" + evaluator.getName()).getValue().floatValue();
            model.setProperty(evaluator.getName(), String.format("%.5f", value));
        }
    }

    public String getTrainingStatus(Trainer trainer) {
        Metrics metrics = trainer.getMetrics();
        StringBuilder sb = new StringBuilder();

        sb.append(metricsOutput(metrics, trainer.getTrainingEvaluators(), "train_"));

        if (metrics.hasMetric("train")) {
            float batchTime = metrics.latestMetric("train").getValue().longValue() / 1_000_000_000f;
            sb.append(String.format(", speed: %.2f images/sec", (float) batchSize / batchTime));
        }
        return sb.toString();
    }

    public void printTrainingStatus(Trainer trainer) {
        Metrics metrics = trainer.getMetrics();
        Loss loss = trainer.getLoss();

        logger.info("Train: " + metricsOutput(metrics, trainer.getTrainingEvaluators(), "train_"));
        if (metrics.hasMetric("validate_" + loss.getName())) {
            logger.info(
                    "Validate: "
                            + metricsOutput(
                                    metrics, trainer.getValidationEvaluators(), "validate_"));
        } else {
            logger.info("validation has not been run.");
        }
    }

    private String metricsOutput(Metrics metrics, List<Evaluator> toOutput, String prefix) {
        List<String> metricOutputs = new ArrayList<>();
        for (Evaluator evaluator : toOutput) {
            float value =
                    metrics.latestMetric(prefix + evaluator.getName()).getValue().floatValue();
            // use .2 precision to avoid new line in progress bar
            metricOutputs.add(String.format("%s: %.2f", evaluator.getName(), value));
        }
        return String.join(", ", metricOutputs);
    }
}
