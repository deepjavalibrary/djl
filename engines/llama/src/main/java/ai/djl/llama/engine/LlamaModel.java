/*
 * Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.llama.engine;

import ai.djl.BaseModel;
import ai.djl.Model;
import ai.djl.llama.jni.LlamaLibrary;
import ai.djl.llama.jni.ModelParameters;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.nn.Blocks;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;

/** {@code LlamaModel} is the llama.cpp implementation of {@link Model}. */
public class LlamaModel extends BaseModel {

    private long handle = -1;

    /**
     * Constructs a new Model on a given device.
     *
     * @param name the model name
     * @param manager the {@link NDManager} to holds the NDArray
     */
    LlamaModel(String name, NDManager manager) {
        super(name);
        this.manager = manager;
        this.manager.setName("llamaModel");
        dataType = DataType.FLOAT32;
    }

    /** {@inheritDoc} */
    @Override
    public void load(Path modelPath, String prefix, Map<String, ?> options) throws IOException {
        setModelDir(modelPath);
        wasLoaded = true;
        if (block != null) {
            throw new UnsupportedOperationException("Llama does not support dynamic blocks");
        }

        if (prefix == null) {
            prefix = modelName;
        }

        // search for .onnx file with prefix, folder name or "model.onnx"
        Path modelFile = findModelFile(prefix, modelDir.toFile().getName(), "model.gguf");
        if (modelFile == null) {
            throw new FileNotFoundException(".gguf file not found in: " + modelPath);
        }

        ModelParameters param = new ModelParameters(options);
        handle = LlamaLibrary.loadModel(modelFile.toString(), param);
        block = Blocks.identityBlock();
    }

    long getHandle() {
        return handle;
    }

    private Path findModelFile(String... prefixes) {
        if (Files.isRegularFile(modelDir)) {
            Path file = modelDir;
            modelDir = modelDir.getParent();
            String fileName = file.toFile().getName();
            if (fileName.endsWith(".gguf")) {
                modelName = fileName.substring(0, fileName.length() - 5);
            } else {
                modelName = fileName;
            }
            return file;
        }
        for (String prefix : prefixes) {
            Path modelFile = modelDir.resolve(prefix);
            if (Files.isRegularFile(modelFile)) {
                return modelFile;
            }
            if (!prefix.endsWith(".gguf")) {
                modelFile = modelDir.resolve(prefix + ".gguf");
                if (Files.isRegularFile(modelFile)) {
                    return modelFile;
                }
            }
        }
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        if (handle == -1) {
            return;
        }
        LlamaLibrary.delete(handle);
        handle = -1;
        super.close();
    }
}
