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
package ai.djl.pytorch.engine;

import ai.djl.Device;
import ai.djl.MalformedModelException;
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
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Function;

public class PtModel implements Model {

    private Path modelDir;
    private Module module;
    private String modelName;
    private DataType dataType;
    private Map<String, String> properties;

    /**
     * Constructs a new Model on a given device.
     *
     * @param device the device the model should be located on
     */
    PtModel(Device device) {
        device = Device.defaultIfNull(device);
        dataType = DataType.FLOAT32;
        properties = new ConcurrentHashMap<>();
    }

    @Override
    public void load(Path modelPath, String modelName, Map<String, String> options)
            throws IOException, MalformedModelException {
        modelDir = modelPath.toAbsolutePath();
        this.modelName = modelName;
        Path modelFile = modelDir.resolve(modelName + ".pt");
        if (Files.notExists(modelFile)) {
            throw new FileNotFoundException(".pt file not found in: " + modelPath);
        }
        this.module = Module.load(modelFile);
    }

    @Override
    public void save(Path modelPath, String modelName) throws IOException {}

    @Override
    public Block getBlock() {
        return null;
    }

    @Override
    public void setBlock(Block block) {}

    @Override
    public String getName() {
        return null;
    }

    @Override
    public String getProperty(String key) {
        return null;
    }

    @Override
    public void setProperty(String key, String value) {}

    @Override
    public NDManager getNDManager() {
        return null;
    }

    @Override
    public Trainer newTrainer(TrainingConfig trainingConfig) {
        return null;
    }

    @Override
    public <I, O> Predictor<I, O> newPredictor(Translator<I, O> translator) {
        return null;
    }

    @Override
    public PairList<String, Shape> describeInput() {
        return null;
    }

    @Override
    public PairList<String, Shape> describeOutput() {
        return null;
    }

    @Override
    public String[] getArtifactNames() {
        return new String[0];
    }

    @Override
    public <T> T getArtifact(String name, Function<InputStream, T> function) throws IOException {
        return null;
    }

    @Override
    public URL getArtifact(String name) throws IOException {
        return null;
    }

    @Override
    public InputStream getArtifactAsStream(String name) throws IOException {
        return null;
    }

    @Override
    public void setDataType(DataType dataType) {}

    @Override
    public DataType getDataType() {
        return null;
    }

    @Override
    public void cast(DataType dataType) {}

    @Override
    public void close() {}
}
