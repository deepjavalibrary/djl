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
package software.amazon.ai.zoo;

import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.nio.file.Path;
import java.util.Map;
import java.util.function.Function;
import software.amazon.ai.Device;
import software.amazon.ai.Model;
import software.amazon.ai.inference.Predictor;
import software.amazon.ai.ndarray.NDManager;
import software.amazon.ai.ndarray.types.DataDesc;
import software.amazon.ai.ndarray.types.DataType;
import software.amazon.ai.nn.Block;
import software.amazon.ai.training.Trainer;
import software.amazon.ai.training.TrainingConfig;
import software.amazon.ai.translate.Translator;

public class ZooModel<I, O> implements Model {

    private Model model;
    private Translator<I, O> translator;

    public ZooModel(Model model, Translator<I, O> translator) {
        this.model = model;
        this.translator = translator;
    }

    @Override
    public void load(Path modelPath, String modelName, Map<String, String> options, Device device) {
        throw new IllegalArgumentException("ZooModel should not be re-loaded.");
    }

    @Override
    public void save(Path modelPath, String modelName) throws IOException {
        model.save(modelPath, modelName);
    }

    @Override
    public Block getBlock() {
        return model.getBlock();
    }

    @Override
    public void setBlock(Block block) {
        model.setBlock(block);
    }

    /** {@inheritDoc} */
    @Override
    public Trainer newTrainer(TrainingConfig trainingConfig) {
        return model.newTrainer(trainingConfig);
    }

    public Predictor<I, O> newPredictor() {
        return newPredictor(translator, null);
    }

    public Predictor<I, O> newPredictor(Device device) {
        return newPredictor(translator, device);
    }

    /** {@inheritDoc} */
    @Override
    public <P, Q> Predictor<P, Q> newPredictor(Translator<P, Q> translator, Device device) {
        return model.newPredictor(translator, device);
    }

    public Translator<I, O> getTranslator() {
        return translator;
    }

    public void quantize() {
        model.cast(DataType.UINT8);
    }

    /** {@inheritDoc} */
    @Override
    public DataDesc[] describeInput() {
        return model.describeInput();
    }

    /** {@inheritDoc} */
    @Override
    public DataDesc[] describeOutput() {
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
    public void cast(DataType dataType) {
        model.cast(dataType);
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        model.close();
    }
}
