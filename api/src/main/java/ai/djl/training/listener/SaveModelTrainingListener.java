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
import java.util.function.Consumer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** A {@link TrainingListener} that saves a model and can save checkpoints. */
public class SaveModelTrainingListener extends TrainingListenerAdapter {

    private static final Logger logger = LoggerFactory.getLogger(SaveModelTrainingListener.class);

    private String outputDir;
    private String overrideModelName;
    private Consumer<Trainer> onSaveModel;
    private int checkpoint;
    private int epoch;

    /**
     * Constructs a {@link SaveModelTrainingListener} using the model's name.
     *
     * @param outputDir the directory to output the checkpointed models in
     */
    public SaveModelTrainingListener(String outputDir) {
        this(outputDir, null, -1);
    }

    /**
     * Constructs a {@link SaveModelTrainingListener}.
     *
     * @param overrideModelName an override model name to save checkpoints with
     * @param outputDir the directory to output the checkpointed models in
     */
    public SaveModelTrainingListener(String outputDir, String overrideModelName) {
        this(outputDir, overrideModelName, -1);
    }

    /**
     * Constructs a {@link SaveModelTrainingListener}.
     *
     * @param overrideModelName an override model name to save checkpoints with
     * @param outputDir the directory to output the checkpointed models in
     * @param checkpoint adds a checkpoint every n epochs
     */
    public SaveModelTrainingListener(String outputDir, String overrideModelName, int checkpoint) {
        this.outputDir = outputDir;
        this.checkpoint = checkpoint;
        if (outputDir == null) {
            throw new IllegalArgumentException(
                    "Can not save checkpoint without specifying an output directory");
        }
        this.overrideModelName = overrideModelName;
    }

    /** {@inheritDoc} */
    @Override
    public void onEpoch(Trainer trainer) {
        epoch++;
        if (outputDir == null) {
            return;
        }

        if (checkpoint > 0 && epoch % checkpoint == 0) {
            // save model at end of each epoch
            saveModel(trainer);
        }
    }

    /** {@inheritDoc} */
    @Override
    public void onTrainingEnd(Trainer trainer) {
        if (checkpoint == -1 || epoch % checkpoint != 0) {
            saveModel(trainer);
        }
    }

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

    /**
     * Returns the checkpoint frequency (or -1 for no checkpointing) in {@link
     * SaveModelTrainingListener}.
     *
     * @return the checkpoint frequency (or -1 for no checkpointing)
     */
    public int getCheckpoint() {
        return checkpoint;
    }

    /**
     * Sets the checkpoint frequency in {@link SaveModelTrainingListener}.
     *
     * @param checkpoint how many epochs between checkpoints (or -1 for no checkpoints)
     */
    public void setCheckpoint(int checkpoint) {
        this.checkpoint = checkpoint;
    }

    /**
     * Sets the callback function on model saving.
     *
     * <p>This allows user to set custom properties to model metadata.
     *
     * @param onSaveModel the callback function on model saving
     */
    public void setSaveModelCallback(Consumer<Trainer> onSaveModel) {
        this.onSaveModel = onSaveModel;
    }

    protected void saveModel(Trainer trainer) {
        Model model = trainer.getModel();
        String modelName = model.getName();
        if (overrideModelName != null) {
            modelName = overrideModelName;
        }
        try {
            model.setProperty("Epoch", String.valueOf(epoch));
            if (onSaveModel != null) {
                onSaveModel.accept(trainer);
            }
            model.save(Paths.get(outputDir), modelName);
        } catch (IOException e) {
            logger.error("Failed to save checkpoint", e);
        }
    }
}
