/*
 * Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import ai.djl.ndarray.types.Shape;

/** {@code AbstractSymbolBlock} is an abstract implementation of {@link SymbolBlock}. */
public abstract class AbstractSymbolBlock extends AbstractBlock implements SymbolBlock {

    /** Constructs a new {@code AbstractSymbolBlock} instance. */
    public AbstractSymbolBlock() {}

    /**
     * Builds an empty block with the given version for parameter serialization.
     *
     * @param version the version to use for parameter serialization.
     */
    public AbstractSymbolBlock(byte version) {
        super(version);
    }

    /** {@inheritDoc} */
    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        throw new UnsupportedOperationException("not implement!");
    }
}
