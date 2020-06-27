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

import ai.djl.metric.Metrics;
import ai.djl.training.Trainer;

/**
 * {@link EpochTrainingListener} that tracks epochs.
 *
 * <p>Adds "epoch" metric with epoch times and saves "epoch" model property with numEpochs
 */
public class EpochTrainingListener extends TrainingListenerAdapter {

    private long epochTime;
    private int numEpochs;

    /** {@inheritDoc} */
    @Override
    public void onEpoch(Trainer trainer) {
        Metrics metrics = trainer.getMetrics();
        if (metrics != null) {
            metrics.addMetric("epoch", System.nanoTime() - epochTime);
        }
        epochTime = System.nanoTime();
        numEpochs++;
    }
    /** {@inheritDoc} */
    @Override
    public void onTrainingBegin(Trainer trainer) {
        epochTime = System.nanoTime();
    }

    /** {@inheritDoc} */
    @Override
    public void onTrainingEnd(Trainer trainer) {
        trainer.getModel().setProperty("Epoch", Integer.toString(numEpochs));
    }

    /**
     * Returns the number of epochs.
     *
     * @return the number of epochs
     */
    public int getNumEpochs() {
        return numEpochs;
    }
}
