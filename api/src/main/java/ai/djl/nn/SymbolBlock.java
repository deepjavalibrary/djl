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
import ai.djl.ndarray.types.Shape;
import ai.djl.util.PairList;

/**
 * {@code SymbolBlock} is a {@link Block} is used to load models that were exported directly from
 * the engine in its native format.
 */
public interface SymbolBlock extends Block {

    /**
     * Creates an empty SymbolBlock instance.
     *
     * @param manager the manager to be applied in the SymbolBlock
     * @return a new Model instance
     */
    static SymbolBlock newInstance(NDManager manager) {
        return manager.getEngine().newSymbolBlock(manager);
    }

    /** Removes the last block in the symbolic graph. */
    default void removeLastBlock() {
        throw new UnsupportedOperationException("not supported");
    }

    /**
     * Returns a {@link PairList} of output names and shapes stored in model file.
     *
     * @return the {@link PairList} of output names, and shapes
     */
    default PairList<String, Shape> describeOutput() {
        throw new UnsupportedOperationException("not supported");
    }
}
