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
import com.amazon.ai.ndarray.types.Shape;
import java.io.File;

public interface Model {

    static Model loadModel(String modelPath) {
        return loadModel(modelPath, -1);
    }

    static Model loadModel(String modelPath, int epoch) {
        File file = new File(modelPath);
        String modelName = file.getName();
        return loadModel(file, modelName, -1);
    }

    static Model loadModel(File modelPath) {
        return loadModel(modelPath, modelPath.getName(), -1);
    }

    static Model loadModel(File modelPath, String modelName) {
        return loadModel(modelPath, modelPath.getName(), -1);
    }

    static Model loadModel(File modelPath, String modelName, int epoch) {
        return Engine.getInstance().loadModel(modelPath, modelName, epoch);
    }

    Block getNetwork();

    Shape getInputShape();

    Shape getOutputShape();

    String[] getLabels();

    String[] getDataNames();

    void setDataNames(String... names);
}
