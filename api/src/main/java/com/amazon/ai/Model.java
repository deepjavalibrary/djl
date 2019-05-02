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
package com.amazon.ai;

import com.amazon.ai.engine.Engine;
import com.amazon.ai.ndarray.types.DataDesc;
import com.amazon.ai.ndarray.types.DataType;
import java.io.File;
import java.io.IOException;

public interface Model {

    static Model loadModel(String modelPath) throws IOException {
        return loadModel(modelPath, -1);
    }

    static Model loadModel(String modelPath, int epoch) throws IOException {
        File file = new File(modelPath);
        String modelName = file.getName();
        return loadModel(file, modelName, -1);
    }

    static Model loadModel(File modelPath) throws IOException {
        return loadModel(modelPath, modelPath.getName(), -1);
    }

    static Model loadModel(File modelPath, String modelName) throws IOException {
        return loadModel(modelPath, modelPath.getName(), -1);
    }

    static Model loadModel(File modelPath, String modelName, int epoch) throws IOException {
        return Engine.getInstance().loadModel(modelPath, modelName, epoch);
    }

    DataDesc[] describeInput();

    DataDesc[] describeOutput();

    String[] getSynset();

    Model cast(DataType dataType);
}
