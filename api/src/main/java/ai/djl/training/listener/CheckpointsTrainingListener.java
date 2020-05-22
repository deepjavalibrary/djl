/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import ai.djl.Model;
import ai.djl.training.Trainer;
import java.io.IOException;
import java.nio.file.Paths;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** A {@link TrainingListener} that saves a model checkpoint after each epoch. */
public class CheckpointsTrainingListener implements TrainingListener {

    private static final Logger logger = LoggerFactory.getLogger(CheckpointsTrainingListener.class);

    private String outputDir;
    private String overrideModelName;
    private int epoch;

    /**
     * Constructs a {@link CheckpointsTrainingListener} using the model's name.
     *
     * @param outputDir the directory to output the checkpointed models in
     */
    public CheckpointsTrainingListener(String outputDir) {
        this(outputDir, null);
    }

    /**
     * Constructs a {@link CheckpointsTrainingListener}.
     *
     * @param overrideModelName an override model name to save checkpoints with
     * @param outputDir the directory to output the checkpointed models in
     */
    public CheckpointsTrainingListener(String outputDir, String overrideModelName) {
        this.outputDir = outputDir;
        if (outputDir == null) {
            throw new IllegalArgumentException(
                    "Can not save checkpoint without specifying an output directory");
        }
        this.overrideModelName = overrideModelName;
    }

    /** {@inheritDoc} */
    @Override
    public void onEpoch(Trainer trainer) {
        if (outputDir == null) {
            return;
        }
        epoch++;
        // save model at end of each epoch
        Model model = trainer.getModel();
        String modelName = model.getName();
        if (overrideModelName != null) {
            modelName = overrideModelName;
        }
        try {
            model.setProperty("Epoch", String.valueOf(epoch));
            model.save(Paths.get(outputDir), modelName);
        } catch (IOException e) {
            logger.error("Failed to save checkpoint", e);
        }
    }

    /** {@inheritDoc} */
    @Override
    public void onTrainingBatch(Trainer trainer, BatchData batchData) {}

    /** {@inheritDoc} */
    @Override
    public void onValidationBatch(Trainer trainer, BatchData batchData) {}

    /** {@inheritDoc} */
    @Override
    public void onTrainingBegin(Trainer trainer) {}

    /** {@inheritDoc} */
    @Override
    public void onTrainingEnd(Trainer trainer) {}

    /**
     * Returns the override model name to save checkpoints with.
     *
     * @return the override model name to save checkpoints with
     */
    public String getOverrideModelName() {
        return overrideModelName;
    }

    /**
     * Sets the override model name to save checkpoints with.
     *
     * @param overrideModelName the override model name to save checkpoints with
     */
    public void setOverrideModelName(String overrideModelName) {
        this.overrideModelName = overrideModelName;
    }
}
