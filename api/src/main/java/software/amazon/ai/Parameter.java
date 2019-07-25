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
package software.amazon.ai;

import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.ndarray.NDManager;
import software.amazon.ai.training.initializer.Initializer;

public class Parameter {

    String name;
    Block block;
    ParameterType type;
    NDManager manager;
    Initializer initializer;
    NDArray array;

    public Parameter(String name, Block block, ParameterType type) {
        this.name = name;
        this.block = block;
        this.type = type;
    }

    public String getName() {
        return name;
    }

    public ParameterType getType() {
        return type;
    }

    public NDArray getArray() {
        if (!isInitialized()) {
            throw new IllegalStateException("The array has not been initialized");
        }
        return array;
    }

    public boolean isInitialized() {
        return array != null;
    }

    public void setInitializer(NDManager manager, Initializer initializer) {
        this.manager = manager;
        this.initializer = initializer;
    }

    public void initialize(NDList inputs) {
        if (isInitialized()) {
            throw new IllegalStateException("This parameter is already initialized");
        }
        array =
                initializer.initialize(
                        manager,
                        block.getParameterShape(name, inputs),
                        inputs.head().getDataType());
    }
}
