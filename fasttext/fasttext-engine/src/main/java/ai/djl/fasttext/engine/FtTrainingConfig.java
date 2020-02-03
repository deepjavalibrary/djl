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
package ai.djl.fasttext.engine;

import ai.djl.Device;
import ai.djl.training.TrainingConfig;
import ai.djl.training.evaluator.Evaluator;
import ai.djl.training.initializer.Initializer;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import java.nio.file.Path;
import java.util.List;

/** An interface that is responsible for holding the configuration required by fastText training. */
public class FtTrainingConfig implements TrainingConfig {

    private FtTrainingMode trainingMode;
    private Path outputDir;
    private String modelName;
    private boolean quantize;

    FtTrainingConfig(Builder builder) {
        trainingMode = builder.trainingMode;
        outputDir = builder.outputDir;
        modelName = builder.modelName;
        quantize = builder.quantize;
    }

    /**
     * Returns the @{link FtTrainingMode}.
     *
     * @return the @{code FtTrainingMode}
     */
    public FtTrainingMode getTrainingMode() {
        return trainingMode;
    }

    /**
     * Returns the output directory.
     *
     * @return the output directory
     */
    public Path getOutputDir() {
        return outputDir;
    }

    /**
     * Return the name of the model.
     *
     * @return the name of the model
     */
    public String getModelName() {
        return modelName;
    }

    /**
     * Returns whether quantize the model on saving.
     *
     * @return whether quantize the model on saving
     */
    public boolean isQuantize() {
        return quantize;
    }

    /** {@inheritDoc} */
    @Override
    public Device[] getDevices() {
        return new Device[0];
    }

    /** {@inheritDoc} */
    @Override
    public Initializer getInitializer() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public Optimizer getOptimizer() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public Loss getLossFunction() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public List<Evaluator> getEvaluators() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public List<TrainingListener> getTrainingListeners() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public int getBatchSize() {
        return 0;
    }

    /**
     * Creates a builder to build a {@code FtTrainingConfig}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /** A builder to construct a {@code FtTrainingConfig}. */
    public static final class Builder {

        FtTrainingMode trainingMode = FtTrainingMode.SUPERVISED;
        Path outputDir;
        String modelName;
        boolean quantize;

        /**
         * Sets the output directory.
         *
         * @param outputDir the output directory
         * @return this builder
         */
        public Builder setOutputDir(Path outputDir) {
            this.outputDir = outputDir;
            return this;
        }

        /**
         * Sets the the name of the model.
         *
         * @param modelName the name of the model
         * @return this builder
         */
        public Builder setModelName(String modelName) {
            this.modelName = modelName;
            return this;
        }

        /**
         * Sets the optional {@link FtTrainingMode}.
         *
         * @param trainingMode the {@code FtTrainingMode}
         * @return this builder
         */
        public Builder optTrainingMode(FtTrainingMode trainingMode) {
            this.trainingMode = trainingMode;
            return this;
        }

        /**
         * Sets the optional quantize.
         *
         * @param quantize whether quantize the model on saving
         * @return this builder
         */
        public Builder optQuantized(boolean quantize) {
            this.quantize = quantize;
            return this;
        }

        /**
         * Builds a new {@code CookingStackExchange}.
         *
         * @return the new {@code CookingStackExchange}
         */
        public FtTrainingConfig build() {
            return new FtTrainingConfig(this);
        }
    }
}
