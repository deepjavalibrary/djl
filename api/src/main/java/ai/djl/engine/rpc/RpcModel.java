/*
 * Copyright 2025 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.engine.rpc;

import ai.djl.BaseModel;
import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.ndarray.types.DataType;
import ai.djl.util.passthrough.PassthroughNDManager;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Map;

/** {@code RpcModel} is an implementation for the {@link Model} deployed on remote model server. */
public class RpcModel extends BaseModel {

    /**
     * Constructs a new Model on a given device.
     *
     * @param name the model name
     */
    RpcModel(String name) {
        super(name);
        manager = PassthroughNDManager.INSTANCE;
        dataType = DataType.FLOAT32;
    }

    /** {@inheritDoc} */
    @Override
    public void load(Path modelPath, String prefix, Map<String, ?> options)
            throws IOException, MalformedModelException {
        setModelDir(modelPath);
        wasLoaded = true;
    }
}
