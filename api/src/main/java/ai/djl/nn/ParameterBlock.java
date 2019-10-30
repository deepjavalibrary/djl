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
package ai.djl.nn;

import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;

/**
 * {@code ParameterBlock} is an abstract implementation of {@link Block}. It is recommended that all
 * {@link Block} classes that have no children extend the {@code ParameterBlock}.
 */
public abstract class ParameterBlock extends AbstractBlock {

    /** {@inheritDoc} */
    @Override
    public Shape[] initialize(NDManager manager, DataType dataType, Shape[] inputShapes) {
        if (!initialized) {
            beforeInitialize(inputShapes);
            for (Parameter parameter : getDirectParameters()) {
                parameter.initialize(manager, dataType, inputShapes);
            }
            initialized = true;
        }
        return getOutputShapes(manager, inputShapes);
    }

    /** {@inheritDoc} */
    @Override
    public final BlockList getChildren() {
        return new BlockList();
    }
}
