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
package ai.djl.onnxruntime.engine;

import ai.djl.BaseModel;
import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.translate.Translator;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;

/**
 * {@code OrtModel} is the ONNX Runtime implementation of {@link Model}.
 *
 * <p>OrtModel contains all the methods in Model to load and process a model. In addition, it
 * provides ONNX Runtime Specific functionality
 */
public class OrtModel extends BaseModel {

    private OrtEnvironment env;

    /**
     * Constructs a new Model on a given device.
     *
     * @param name the model name
     * @param manager the {@link NDManager} to holds the NDArray
     * @param env the {@link OrtEnvironment} ONNX Environment to create session
     */
    OrtModel(String name, NDManager manager, OrtEnvironment env) {
        super(name);
        this.manager = manager;
        this.env = env;
        dataType = DataType.FLOAT32;
    }

    /** {@inheritDoc} */
    @Override
    public void load(Path modelPath, String prefix, Map<String, Object> options)
            throws IOException, MalformedModelException {
        modelDir = modelPath.toAbsolutePath();
        if (prefix == null) {
            prefix = modelName;
        }
        if (block != null) {
            throw new UnsupportedOperationException("ONNX Runtime does not support dynamic blocks");
        }
        Path modelFile = findModelFile(prefix);
        if (modelFile == null) {
            modelFile = findModelFile(modelDir.toFile().getName());
            if (modelFile == null) {
                throw new FileNotFoundException(".onnx file not found in: " + modelPath);
            }
        }
        // TODO: Support SessionOption here for further optimization
        try {
            block = new OrtSymbolBlock(env.createSession(modelFile.toString()));
        } catch (OrtException e) {
            throw new MalformedModelException("ONNX Model cannot be loaded", e);
        }
    }

    private Path findModelFile(String prefix) {
        Path modelFile = modelDir.resolve(prefix);
        if (Files.notExists(modelFile) || !Files.isRegularFile(modelFile)) {
            if (prefix.endsWith(".onnx")) {
                return null;
            }
            modelFile = modelDir.resolve(prefix + ".onnx");
            if (Files.notExists(modelFile) || !Files.isRegularFile(modelFile)) {
                return null;
            }
        }
        return modelFile;
    }

    /** {@inheritDoc} */
    @Override
    public Trainer newTrainer(TrainingConfig trainingConfig) {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public <I, O> Predictor<I, O> newPredictor(Translator<I, O> translator) {
        return new Predictor<>(this, translator, false);
    }

    /** {@inheritDoc} */
    @Override
    public String[] getArtifactNames() {
        return new String[0];
    }

    /** {@inheritDoc} */
    @Override
    public void cast(DataType dataType) {
        throw new UnsupportedOperationException("Not supported for ONNX Runtime");
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        manager.close();
    }
}
