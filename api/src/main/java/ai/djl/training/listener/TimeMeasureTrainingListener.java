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

import ai.djl.metric.Metric;
import ai.djl.metric.Metrics;
import ai.djl.training.Trainer;
import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * {@link TrainingListener} that outputs the training time metrics after training is done.
 *
 * <p>The training time data is placed in the file "$outputDir/training.log" and the validation data
 * is placed in "$outputDir/validate.log".
 */
public class TimeMeasureTrainingListener extends TrainingListenerAdapter {

    private static final Logger logger = LoggerFactory.getLogger(TimeMeasureTrainingListener.class);

    private String outputDir;
    private long trainBatchBeginTime;
    private long validateBatchBeginTime;

    /**
     * Constructs a {@link TimeMeasureTrainingListener}.
     *
     * @param outputDir the directory to output the tracked timing data in
     */
    public TimeMeasureTrainingListener(String outputDir) {
        this.outputDir = outputDir;
        trainBatchBeginTime = -1;
        validateBatchBeginTime = -1;
    }

    /** {@inheritDoc} */
    @Override
    public void onEpoch(Trainer trainer) {
        trainBatchBeginTime = -1;
        validateBatchBeginTime = -1;
    }

    /** {@inheritDoc} */
    @Override
    public void onTrainingBatch(Trainer trainer, BatchData batchData) {
        if (trainBatchBeginTime != -1) {
            trainer.addMetric("train", trainBatchBeginTime);
        }
        trainBatchBeginTime = System.nanoTime();
    }

    /** {@inheritDoc} */
    @Override
    public void onValidationBatch(Trainer trainer, BatchData batchData) {
        if (validateBatchBeginTime != -1) {
            trainer.addMetric("validate", validateBatchBeginTime);
        }
        validateBatchBeginTime = System.nanoTime();
    }

    /** {@inheritDoc} */
    @Override
    public void onTrainingEnd(Trainer trainer) {
        Metrics metrics = trainer.getMetrics();
        dumpTrainingTimeInfo(metrics, outputDir);
    }

    private static void dumpTrainingTimeInfo(Metrics metrics, String logDir) {
        if (metrics == null || logDir == null) {
            return;
        }
        try {
            Path dir = Paths.get(logDir);
            Files.createDirectories(dir);
            dumpMetricToFile(dir.resolve("training.log"), metrics.getMetric("train"));
            dumpMetricToFile(dir.resolve("validate.log"), metrics.getMetric("validate"));
        } catch (IOException e) {
            logger.error("Failed dump training log", e);
        }
    }

    private static void dumpMetricToFile(Path path, List<Metric> metrics) throws IOException {
        if (metrics == null || metrics.isEmpty()) {
            return;
        }

        try (BufferedWriter writer =
                Files.newBufferedWriter(
                        path, StandardOpenOption.CREATE, StandardOpenOption.APPEND)) {
            for (Metric metric : metrics) {
                writer.append(metric.toString());
                writer.newLine();
            }
        }
    }
}
