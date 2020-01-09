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

import ai.djl.examples.util.MemoryUtils;
import ai.djl.metric.Metrics;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingListener;

public class MemoryTrainingListener implements TrainingListener {

    private String outputDir;

    public MemoryTrainingListener() {}

    public MemoryTrainingListener(String outputDir) {
        this.outputDir = outputDir;
    }

    @Override
    public void onEpoch(Trainer trainer) {}

    @Override
    public void onTrainingBatch(Trainer trainer) {
        Metrics metrics = trainer.getMetrics();
        MemoryUtils.collectMemoryInfo(metrics);
    }

    @Override
    public void onValidationBatch(Trainer trainer) {
        Metrics metrics = trainer.getMetrics();
        MemoryUtils.collectMemoryInfo(metrics);
    }

    @Override
    public void onTrainingBegin(Trainer trainer) {}

    @Override
    public void onTrainingEnd(Trainer trainer) {
        Metrics metrics = trainer.getMetrics();
        if (outputDir != null) {
            MemoryUtils.dumpMemoryInfo(metrics, outputDir);
        }
    }
}
