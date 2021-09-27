/*
 * Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.tensorrt.engine;

import ai.djl.BaseModel;
import ai.djl.Model;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.types.DataType;
import ai.djl.tensorrt.jni.JniUtils;
import ai.djl.translate.Translator;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;

/**
 * {@code TrtModel} is the TensorRT implementation of {@link Model}.
 *
 * <p>OrtModel contains all the methods in Model to load and process a model. In addition, it
 * provides TensorRT Specific functionality
 */
public class TrtModel extends BaseModel {

    /**
     * Constructs a new Model on a given device.
     *
     * @param name the model name
     * @param manager the {@link TrtNDManager} to holds the NDArray
     */
    TrtModel(String name, TrtNDManager manager) {
        super(name);
        this.manager = manager;
        this.manager.setName("tensorrtModel");
        dataType = DataType.FLOAT32;
    }

    /** {@inheritDoc} */
    @Override
    public void load(Path modelPath, String prefix, Map<String, ?> options) throws IOException {
        if (block != null) {
            throw new UnsupportedOperationException("TensorRT does not support dynamic blocks");
        }
        modelDir = modelPath.toAbsolutePath();
        if (prefix == null) {
            prefix = modelName;
        }
        Path modelFile = findModelFile(prefix);
        if (modelFile == null) {
            modelFile = findModelFile(modelDir.toFile().getName());
            if (modelFile == null) {
                throw new FileNotFoundException(prefix + ".* file not found in: " + modelDir);
            }
        }
        String filePath = modelFile.toString();
        int modelType;
        if (filePath.endsWith(".onnx")) {
            modelType = 0;
        } else if (filePath.endsWith(".uff")) {
            modelType = 1;
        } else {
            modelType = 2;
        }

        long modelHandle = JniUtils.loadModel(modelType, filePath, manager.getDevice(), options);
        block = new TrtSymbolBlock(modelHandle);
    }

    /** {@inheritDoc} */
    @Override
    public <I, O> Predictor<I, O> newPredictor(Translator<I, O> translator) {
        TrtSymbolBlock trtSymbol = ((TrtSymbolBlock) block);
        TrtSession session = trtSymbol.createSession((TrtNDManager) manager);
        return new TrtPredictor<>(this, translator, session);
    }

    private Path findModelFile(String prefix) {
        if (Files.isRegularFile(modelDir)) {
            Path file = modelDir;
            modelDir = modelDir.getParent();
            String fileName = file.toFile().getName();
            if (fileName.endsWith(".onnx")) {
                modelName = fileName.substring(0, fileName.length() - 5);
            } else if (fileName.endsWith(".trt") || fileName.endsWith(".uff")) {
                modelName = fileName.substring(0, fileName.length() - 4);
            } else {
                modelName = fileName;
            }
            return file;
        }
        Path modelFile = modelDir.resolve(prefix);
        if (Files.notExists(modelFile) || !Files.isRegularFile(modelFile)) {
            if (prefix.endsWith(".onnx") || prefix.endsWith(".trt") || prefix.endsWith(".uff")) {
                return null;
            }
            modelFile = modelDir.resolve(prefix + ".onnx");
            if (Files.notExists(modelFile) || !Files.isRegularFile(modelFile)) {
                modelFile = modelDir.resolve(prefix + ".trt");
                if (Files.notExists(modelFile) || !Files.isRegularFile(modelFile)) {
                    modelFile = modelDir.resolve(prefix + ".uff");
                    if (Files.notExists(modelFile) || !Files.isRegularFile(modelFile)) {
                        return null;
                    }
                }
            }
        }
        return modelFile;
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        if (block != null) {
            ((TrtSymbolBlock) block).close();
            block = null;
        }
        super.close();
    }
}
