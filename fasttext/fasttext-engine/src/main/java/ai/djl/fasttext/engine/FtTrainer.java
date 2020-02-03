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
import ai.djl.Model;
import ai.djl.fasttext.dataset.FtDataset;
import ai.djl.metric.Metrics;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.GradientCollector;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.training.dataset.Batch;
import ai.djl.training.evaluator.Evaluator;
import ai.djl.training.loss.Loss;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import org.bytedeco.javacpp.CharPointer;
import org.bytedeco.javacpp.PointerPointer;

/** {@code FtTrainer} is the fastText implementation of the {@link Trainer}. */
public class FtTrainer implements Trainer {

    private FtModel model;
    private FtTrainingConfig config;
    private Metrics metrics;

    /**
     * Creates an instance of {@code FtTrainer} with the given {@link FtModel} and {@link
     * TrainingConfig}.
     *
     * @param model the model the trainer will train on
     * @param trainingConfig the configuration used by the trainer
     */
    FtTrainer(FtModel model, TrainingConfig trainingConfig) {
        this.model = model;
        this.config = (FtTrainingConfig) trainingConfig;
    }

    /**
     * Train the fastText model.
     *
     * @param trainingSet the training dataset
     * @param validateSet the validation dataset
     * @throws IOException when IO operation fails in loading a resource
     */
    public void fit(FtDataset trainingSet, FtDataset validateSet) throws IOException {
        Path outputDir = config.getOutputDir();
        if (Files.notExists(outputDir)) {
            Files.createDirectory(outputDir);
        }
        String modelName = config.getModelName();
        Path modelFile = outputDir.resolve(modelName).toAbsolutePath();

        List<String> cmd = new ArrayList<>(6);
        cmd.add("fasttext");
        cmd.add(config.getTrainingMode().name().toLowerCase());
        cmd.add("-input");
        cmd.add(trainingSet.getInputFile().toString());
        cmd.add("-output");
        cmd.add(modelFile.toString());
        String[] args = cmd.toArray(new String[0]);

        model.fta.runCmd(args.length, new PointerPointer<CharPointer>(args));
        model.setModelFile(modelFile);
    }

    /** {@inheritDoc} */
    @Override
    public void initialize(Shape... shapes) {}

    /** {@inheritDoc} */
    @Override
    public GradientCollector newGradientCollector() {
        throw new UnsupportedOperationException("Fasttest doesn't support AutoGrad");
    }

    /** {@inheritDoc} */
    @Override
    public void trainBatch(Batch batch) {}

    /** {@inheritDoc} */
    @Override
    public NDList forward(NDList input) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public void validateBatch(Batch batch) {}

    /** {@inheritDoc} */
    @Override
    public void step() {}

    /** {@inheritDoc} */
    @Override
    public Metrics getMetrics() {
        return metrics;
    }

    /** {@inheritDoc} */
    @Override
    public void setMetrics(Metrics metrics) {
        this.metrics = metrics;
    }

    /** {@inheritDoc} */
    @Override
    public List<Device> getDevices() {
        return Collections.emptyList();
    }

    /** {@inheritDoc} */
    @Override
    public void endEpoch() {}

    /** {@inheritDoc} */
    @Override
    public Loss getLoss() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public Model getModel() {
        return model;
    }

    /** {@inheritDoc} */
    @Override
    public List<Evaluator> getEvaluators() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public final <T extends Evaluator> T getEvaluator(Class<T> clazz) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDManager getManager() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public void close() {}
}
