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
package ai.djl.paddlepaddle.engine;

import ai.djl.BaseModel;
import ai.djl.Model;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.paddlepaddle.jna.JnaUtils;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;

/** {@code PpModel} is the PaddlePaddle implementation of {@link Model}. */
public class PpModel extends BaseModel {

    private AnalysisConfig config;

    /**
     * Constructs a new Model on a given device.
     *
     * @param name the model name
     */
    PpModel(String name, NDManager manager) {
        super(name);
        this.manager = manager;
        config = new AnalysisConfig(manager.getDevice());
        dataType = DataType.FLOAT32;
        manager.setName("PpModel");
    }

    /**
     * Loads the PaddlePaddle model from a specified location.
     *
     * <pre>
     * Map&lt;String, String&gt; options = new HashMap&lt;&gt;()
     * <b>options.put("epoch", "3");</b>
     * model.load(modelPath, "squeezenet", options);
     * </pre>
     *
     * @param modelPath the directory of the model
     * @param prefix the model file name or path prefix
     * @param options load model options, see documentation for the specific engine
     * @throws IOException Exception for file loading
     */
    @Override
    public void load(Path modelPath, String prefix, Map<String, ?> options) throws IOException {
        modelDir = modelPath.toAbsolutePath();
        Path modelFile = modelDir.resolve("model");
        if (Files.exists(modelFile)) {
            // TODO: Allow only files in the zip folder
            if (Files.isDirectory(modelFile)) {
                load(modelPath.resolve("model"), prefix, options);
                return;
            }
            Path paramFile = modelDir.resolve("params");
            if (Files.notExists(paramFile)) {
                throw new FileNotFoundException("params file not found in: " + modelDir);
            }
            JnaUtils.setModel(config, modelFile.toString(), paramFile.toString());
            setBlock(new PpSymbolBlock(config));
            return;
        }

        modelFile = modelDir.resolve("__model__");
        if (Files.notExists(modelFile)) {
            throw new FileNotFoundException("no __model__ or model file found in: " + modelDir);
        }
        JnaUtils.setModel(config, modelDir.toString(), null);
        setBlock(new PpSymbolBlock(config));
    }

    /** {@inheritDoc} */
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder(200);
        sb.append("Model (\n\tName: ").append(modelName);
        if (modelDir != null) {
            sb.append("\n\tModel location: ").append(modelDir.toAbsolutePath());
        }
        sb.append("\n\tData Type: ").append(dataType);
        for (Map.Entry<String, String> entry : properties.entrySet()) {
            sb.append("\n\t").append(entry.getKey()).append(": ").append(entry.getValue());
        }
        sb.append("\n)");
        return sb.toString();
    }
}
