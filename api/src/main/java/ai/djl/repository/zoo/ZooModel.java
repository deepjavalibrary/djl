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
package ai.djl.repository.zoo;

import ai.djl.Model;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.translate.Translator;
import ai.djl.util.PairList;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.nio.file.Path;
import java.util.Map;
import java.util.function.Function;

/**
 * A {@code ZooModel} is a {@link Model} loaded from a model zoo and includes a default {@link
 * Translator}.
 *
 * @param <I> the model input type
 * @param <O> the model output type
 */
public class ZooModel<I, O> implements Model {

    private Model model;
    private Translator<I, O> translator;

    /**
     * Constructs a {@code ZooModel} given the model and translator.
     *
     * @param model the model to wrap
     * @param translator the translator
     */
    public ZooModel(Model model, Translator<I, O> translator) {
        this.model = model;
        this.translator = translator;
    }

    /** {@inheritDoc} */
    @Override
    public void load(Path modelPath, String prefix, Map<String, ?> options) {
        throw new IllegalArgumentException("ZooModel should not be re-loaded.");
    }

    /**
     * Returns the wrapped model.
     *
     * @return the wrapped model
     */
    public Model getWrappedModel() {
        return model;
    }

    /** {@inheritDoc} */
    @Override
    public void save(Path modelPath, String modelName) throws IOException {
        model.save(modelPath, modelName);
    }

    /** {@inheritDoc} */
    @Override
    public Path getModelPath() {
        return model.getModelPath();
    }

    /** {@inheritDoc} */
    @Override
    public Block getBlock() {
        return model.getBlock();
    }

    /** {@inheritDoc} */
    @Override
    public void setBlock(Block block) {
        model.setBlock(block);
    }

    /** {@inheritDoc} */
    @Override
    public String getName() {
        return model.getName();
    }

    /** {@inheritDoc} */
    @Override
    public String getProperty(String key) {
        return model.getProperty(key);
    }

    /** {@inheritDoc} */
    @Override
    public void setProperty(String key, String value) {
        model.setProperty(key, value);
    }

    /** {@inheritDoc} */
    @Override
    public Trainer newTrainer(TrainingConfig trainingConfig) {
        return model.newTrainer(trainingConfig);
    }

    /**
     * Creates a new Predictor based on the model with the default translator.
     *
     * @return an instance of {@code Predictor}
     */
    public Predictor<I, O> newPredictor() {
        return newPredictor(translator);
    }

    /** {@inheritDoc} */
    @Override
    public <P, Q> Predictor<P, Q> newPredictor(Translator<P, Q> translator) {
        return model.newPredictor(translator);
    }

    /**
     * Returns the default translator.
     *
     * @return the default translator
     */
    public Translator<I, O> getTranslator() {
        return translator;
    }

    /** {@inheritDoc} */
    @Override
    public PairList<String, Shape> describeInput() {
        return model.describeInput();
    }

    /** {@inheritDoc} */
    @Override
    public PairList<String, Shape> describeOutput() {
        return model.describeOutput();
    }

    /** {@inheritDoc} */
    @Override
    public String[] getArtifactNames() {
        return model.getArtifactNames();
    }

    /** {@inheritDoc} */
    @Override
    public <T> T getArtifact(String name, Function<InputStream, T> function) throws IOException {
        return model.getArtifact(name, function);
    }

    /** {@inheritDoc} */
    @Override
    public URL getArtifact(String name) throws IOException {
        return model.getArtifact(name);
    }

    /** {@inheritDoc} */
    @Override
    public InputStream getArtifactAsStream(String name) throws IOException {
        return model.getArtifactAsStream(name);
    }

    /** {@inheritDoc} */
    @Override
    public NDManager getNDManager() {
        return model.getNDManager();
    }

    /** {@inheritDoc} */
    @Override
    public void setDataType(DataType dataType) {
        model.setDataType(dataType);
    }

    /** {@inheritDoc} */
    @Override
    public DataType getDataType() {
        return model.getDataType();
    }

    /** {@inheritDoc} */
    @Override
    public void cast(DataType dataType) {
        model.cast(dataType);
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        model.close();
    }
}
