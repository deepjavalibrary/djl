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
package ai.djl.pytorch.engine;

import ai.djl.BaseModel;
import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.pytorch.jni.JniUtils;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.translate.Translator;
import ai.djl.util.PairList;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * {@code PtModel} is the PyTorch implementation of {@link Model}.
 *
 * <p>PtModel contains all the methods in Model to load and process a model. In addition, it
 * provides PyTorch Specific functionality
 */
public class PtModel extends BaseModel {

    PtNDManager manager;
    /**
     * Constructs a new Model on a given device.
     *
     * @param device the device the model should be located on
     */
    PtModel(Device device) {
        device = Device.defaultIfNull(device);
        manager = PtNDManager.getSystemManager().newSubManager(device);
    }

    @Override
    public void load(Path modelPath, String modelName, Map<String, String> options)
            throws IOException, MalformedModelException {
        modelDir = modelPath.toAbsolutePath();
        Path modelFile = modelDir.resolve(modelName + ".pt");
        if (Files.notExists(modelFile)) {
            throw new FileNotFoundException(".pt file not found in: " + modelPath);
        }
        block = JniUtils.loadModule(manager, modelFile);
    }

    @Override
    public void save(Path modelPath, String modelName) throws IOException {}

    @Override
    public PtNDManager getNDManager() {
        return manager;
    }

    @Override
    public Trainer newTrainer(TrainingConfig trainingConfig) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public <I, O> Predictor<I, O> newPredictor(Translator<I, O> translator) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public PairList<String, Shape> describeInput() {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public PairList<String, Shape> describeOutput() {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public String[] getArtifactNames() {
        try {
            List<Path> files =
                    Files.walk(modelDir).filter(Files::isRegularFile).collect(Collectors.toList());
            List<String> ret = new ArrayList<>(files.size());
            for (Path path : files) {
                String fileName = path.toFile().getName();
                if (fileName.endsWith(".pt")) {
                    // ignore model files.
                    continue;
                }
                Path relative = modelDir.relativize(path);
                ret.add(relative.toString());
            }
            return ret.toArray(new String[0]);
        } catch (IOException e) {
            throw new AssertionError("Failed list files", e);
        }
    }

    @Override
    public void setDataType(DataType dataType) {}

    @Override
    public DataType getDataType() {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public void cast(DataType dataType) {}

    @Override
    public void close() {}
}
