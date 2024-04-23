/*
 * Copyright 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.engine.rust;

import ai.djl.BaseModel;
import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.ndarray.types.DataType;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;

/** {@code RsModel} is the Rust implementation of {@link Model}. */
public class RsModel extends BaseModel {

    /**
     * Constructs a new Model on a given device.
     *
     * @param name the model name
     * @param device the device the model should be located on
     */
    RsModel(String name, Device device) {
        super(name);
        manager = RsNDManager.getSystemManager().newSubManager(device);
        manager.setName("RsModel");
        dataType = DataType.FLOAT16;
    }

    /** {@inheritDoc} */
    @Override
    public void load(Path modelPath, String prefix, Map<String, ?> options)
            throws IOException, MalformedModelException {
        if (Files.notExists(modelPath)) {
            throw new FileNotFoundException(
                    "Model directory doesn't exist: " + modelPath.toAbsolutePath());
        }
        modelDir = modelPath.toAbsolutePath();
        long handle = RustLibrary.loadModel(modelDir.toString(), dataType.ordinal());
        block = new RsSymbolBlock((RsNDManager) manager, handle);
    }
}
