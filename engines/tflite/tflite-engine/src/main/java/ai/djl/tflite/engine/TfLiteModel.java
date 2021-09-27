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
package ai.djl.tflite.engine;

import ai.djl.BaseModel;
import ai.djl.Model;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import org.tensorflow.lite.Interpreter;

/**
 * {@code TfLiteModel} is the TFLite implementation of {@link Model}.
 *
 * <p>TfLiteModel contains all the methods in Model to load and process a model. In addition, it
 * provides TFLite Specific functionality
 */
public class TfLiteModel extends BaseModel {

    /**
     * Constructs a new Model on a given device.
     *
     * @param name the model name
     * @param manager the {@link NDManager} to holds the NDArray
     */
    TfLiteModel(String name, NDManager manager) {
        super(name);
        this.manager = TfLiteNDManager.getSystemManager().newSubManager();
        manager.setName("TfLiteModel");
        dataType = DataType.FLOAT32;
    }

    /** {@inheritDoc} */
    @Override
    public void load(Path modelPath, String prefix, Map<String, ?> options) throws IOException {
        modelDir = modelPath.toAbsolutePath();
        if (block != null) {
            throw new UnsupportedOperationException("TFLite does not support dynamic blocks");
        }
        Path modelFile = findModelFile(prefix);
        if (modelFile == null) {
            modelFile = findModelFile(modelDir.toFile().getName());
            if (modelFile == null) {
                throw new FileNotFoundException("TFLite model file not found in: " + modelPath);
            }
        }
        Interpreter interpreter = new Interpreter(modelFile.toFile());
        setBlock(new TfLiteSymbolBlock(interpreter, getNDManager()));
    }

    /** {@inheritDoc} */
    @Override
    public TfLiteNDManager getNDManager() {
        return (TfLiteNDManager) super.getNDManager();
    }

    private Path findModelFile(String prefix) {
        if (Files.isRegularFile(modelDir)) {
            Path file = modelDir;
            modelDir = modelDir.getParent();
            String fileName = file.toFile().getName();
            if (fileName.endsWith(".tflite")) {
                modelName = fileName.substring(0, fileName.length() - 5);
            } else {
                modelName = fileName;
            }
            return file;
        }
        if (prefix == null) {
            prefix = modelName;
        }
        Path modelFile = modelDir.resolve(prefix);
        if (Files.notExists(modelFile) || !Files.isRegularFile(modelFile)) {
            if (prefix.endsWith(".tflite")) {
                return null;
            }
            modelFile = modelDir.resolve(prefix + ".tflite");
            if (Files.notExists(modelFile) || !Files.isRegularFile(modelFile)) {
                return null;
            }
        }
        return modelFile;
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        if (block != null) {
            ((TfLiteSymbolBlock) block).close();
            block = null;
        }
        super.close();
    }
}
