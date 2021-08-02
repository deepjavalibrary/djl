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
package ai.djl.training.listener;

import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.metric.Metrics;
import ai.djl.training.Trainer;
import ai.djl.training.evaluator.Evaluator;
import ai.djl.training.loss.Loss;
import ai.djl.training.util.ProgressBar;
import java.util.ArrayList;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * {@link TrainingListener} that outputs the progress of training each batch and epoch into logs.
 *
 * @see <a href="http://docs.djl.ai/docs/development/configure_logging.html">The guide on DJL
 *     logging</a>
 */
public class LoggingTrainingListener implements TrainingListener {

    private static final Logger logger = LoggerFactory.getLogger(LoggingTrainingListener.class);

    private int frequency;

    private int numEpochs;
    private ProgressBar trainingProgressBar;
    private ProgressBar validateProgressBar;

    /** Constructs a {@code LoggingTrainingListener} instance. */
    public LoggingTrainingListener() {}

    /**
     * Constructs a {@code LoggingTrainingListener} instance with specified steps.
     *
     * <p>Print out logs every {@code frequency} epoch.
     *
     * @param frequency the frequency of epoch to print out
     */
    public LoggingTrainingListener(int frequency) {
        this.frequency = frequency;
    }

    /** {@inheritDoc} */
    @Override
    public void onEpoch(Trainer trainer) {
        numEpochs++;
        if (frequency > 1 && numEpochs % frequency != 1) {
            return;
        }

        logger.info("Epoch {} finished.", numEpochs);

        Metrics metrics = trainer.getMetrics();
        if (metrics != null) {
            Loss loss = trainer.getLoss();
            String status =
                    getEvaluatorsStatus(
                            metrics,
                            trainer.getEvaluators(),
                            EvaluatorTrainingListener.TRAIN_EPOCH,
                            Short.MAX_VALUE);
            logger.info("Train: {}", status);

            String metricName =
                    EvaluatorTrainingListener.metricName(
                            loss, EvaluatorTrainingListener.VALIDATE_EPOCH);
            if (metrics.hasMetric(metricName)) {
                status =
                        getEvaluatorsStatus(
                                metrics,
                                trainer.getEvaluators(),
                                EvaluatorTrainingListener.VALIDATE_EPOCH,
                                Short.MAX_VALUE);
                if (!status.isEmpty()) {
                    logger.info("Validate: {}", status);
                }
            } else {
                logger.info("validation has not been run.");
            }
        }
    }

    /** {@inheritDoc} */
    @Override
    public void onTrainingBatch(Trainer trainer, BatchData batchData) {
        if (frequency > 1 && numEpochs % frequency != 1) {
            return;
        }

        if (trainingProgressBar == null) {
            trainingProgressBar =
                    new ProgressBar("Training", batchData.getBatch().getProgressTotal());
        }
        trainingProgressBar.update(
                batchData.getBatch().getProgress(),
                getTrainingStatus(trainer, batchData.getBatch().getSize()));
    }

    private String getTrainingStatus(Trainer trainer, int batchSize) {
        Metrics metrics = trainer.getMetrics();
        if (metrics == null) {
            return "";
        }

        StringBuilder sb = new StringBuilder();
        sb.append(
                getEvaluatorsStatus(
                        metrics,
                        trainer.getEvaluators(),
                        EvaluatorTrainingListener.TRAIN_PROGRESS,
                        2));

        if (metrics.hasMetric("train")) {
            float batchTime = metrics.latestMetric("train").getValue().longValue() / 1_000_000_000f;
            sb.append(String.format(", speed: %.2f items/sec", batchSize / batchTime));
        }
        return sb.toString();
    }

    /** {@inheritDoc} */
    @Override
    public void onValidationBatch(Trainer trainer, BatchData batchData) {
        if (frequency > 1 && numEpochs % frequency != 1) {
            return;
        }

        if (validateProgressBar == null) {
            validateProgressBar =
                    new ProgressBar("Validating", batchData.getBatch().getProgressTotal());
        }
        validateProgressBar.update(batchData.getBatch().getProgress());
    }

    /** {@inheritDoc} */
    @Override
    public void onTrainingBegin(Trainer trainer) {
        String devicesMsg;
        Device[] devices = trainer.getDevices();
        if (devices.length == 1 && Device.Type.CPU.equals(devices[0].getDeviceType())) {
            devicesMsg = Device.cpu().toString();
        } else {
            devicesMsg = devices.length + " GPUs";
        }
        logger.info("Training on: {}.", devicesMsg);

        long init = System.nanoTime();
        Engine engine = trainer.getManager().getEngine();
        String engineName = engine.getEngineName();
        String version = engine.getVersion();
        long loaded = System.nanoTime();
        logger.info(
                String.format(
                        "Load %s Engine Version %s in %.3f ms.",
                        engineName, version, (loaded - init) / 1_000_000f));
    }

    /** {@inheritDoc} */
    @Override
    public void onTrainingEnd(Trainer trainer) {
        Metrics metrics = trainer.getMetrics();
        if (metrics == null) {
            return;
        }

        float p50;
        float p90;
        if (metrics.hasMetric("train")) {
            // possible no train metrics if only one iteration is executed
            p50 = metrics.percentile("train", 50).getValue().longValue() / 1_000_000f;
            p90 = metrics.percentile("train", 90).getValue().longValue() / 1_000_000f;
            logger.info(String.format("train P50: %.3f ms, P90: %.3f ms", p50, p90));
        }

        if (metrics.hasMetric("forward")) {
            p50 = metrics.percentile("forward", 50).getValue().longValue() / 1_000_000f;
            p90 = metrics.percentile("forward", 90).getValue().longValue() / 1_000_000f;
            logger.info(String.format("forward P50: %.3f ms, P90: %.3f ms", p50, p90));
        }

        if (metrics.hasMetric("training-metrics")) {
            p50 = metrics.percentile("training-metrics", 50).getValue().longValue() / 1_000_000f;
            p90 = metrics.percentile("training-metrics", 90).getValue().longValue() / 1_000_000f;
            logger.info(String.format("training-metrics P50: %.3f ms, P90: %.3f ms", p50, p90));
        }

        if (metrics.hasMetric("backward")) {
            p50 = metrics.percentile("backward", 50).getValue().longValue() / 1_000_000f;
            p90 = metrics.percentile("backward", 90).getValue().longValue() / 1_000_000f;
            logger.info(String.format("backward P50: %.3f ms, P90: %.3f ms", p50, p90));
        }

        if (metrics.hasMetric("step")) {
            p50 = metrics.percentile("step", 50).getValue().longValue() / 1_000_000f;
            p90 = metrics.percentile("step", 90).getValue().longValue() / 1_000_000f;
            logger.info(String.format("step P50: %.3f ms, P90: %.3f ms", p50, p90));
        }

        if (metrics.hasMetric("epoch")) {
            p50 = metrics.percentile("epoch", 50).getValue().longValue() / 1_000_000_000f;
            p90 = metrics.percentile("epoch", 90).getValue().longValue() / 1_000_000_000f;
            logger.info(String.format("epoch P50: %.3f s, P90: %.3f s", p50, p90));
        }
    }

    private String getEvaluatorsStatus(
            Metrics metrics, List<Evaluator> toOutput, String stage, int limit) {
        List<String> metricOutputs = new ArrayList<>(limit + 1);
        int count = 0;
        for (Evaluator evaluator : toOutput) {
            if (++count > limit) {
                metricOutputs.add("...");
                break;
            }
            String metricName = EvaluatorTrainingListener.metricName(evaluator, stage);
            if (metrics.hasMetric(metricName)) {
                float value = metrics.latestMetric(metricName).getValue().floatValue();
                // use .2 precision to avoid new line in progress bar
                String output;
                if (Math.abs(value) < .01 || Math.abs(value) > 9999) {
                    output = String.format("%s: %.2E", evaluator.getName(), value);
                } else if (metricName.startsWith("validate_") && Float.isNaN(value)) {
                    continue;
                } else {
                    output = String.format("%s: %.2f", evaluator.getName(), value);
                }
                metricOutputs.add(output);
            } else {
                metricOutputs.add(String.format("%s: _", evaluator.getName()));
            }
        }
        return String.join(", ", metricOutputs);
    }
}
