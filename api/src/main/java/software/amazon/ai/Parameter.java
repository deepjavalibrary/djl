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

import java.util.Objects;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.ndarray.NDManager;
import software.amazon.ai.training.initializer.Initializer;

public class Parameter implements AutoCloseable {

    private String name;
    private Block block;
    private ParameterType type;
    private NDManager manager;
    private Initializer initializer;
    private NDArray array;
    private boolean gradientAttached;

    public Parameter(String name, Block block, ParameterType type) {
        this.name = name;
        this.block = block;
        this.type = type;
    }

    public Parameter(String name, NDArray array) {
        this.name = name;
        this.array = array;
    }

    public void attachGradient() {
        if (!isInitialized()) {
            throw new IllegalStateException("This parameter is not initialized");
        }
        if (!gradientAttached) {
            array.attachGradient();
            gradientAttached = true;
        }
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

    public void reinitialize() {
        if (!isInitialized()) {
            throw new IllegalStateException("This parameter is not initialized");
        }
        Objects.requireNonNull(initializer, "No initializer has been set");
        array = initializer.initialize(manager, array.getShape(), array.getDataType());
        attachGradient();
    }

    public void initialize(NDList inputs) {
        if (isInitialized()) {
            throw new IllegalStateException("This parameter is already initialized");
        }

        Objects.requireNonNull(initializer, "No initializer has been set");
        array =
                initializer.initialize(
                        manager,
                        block.getParameterShape(name, inputs),
                        inputs.head().getDataType());
        attachGradient();
    }

    @Override
    public void close() {
        array.close();
    }
}
