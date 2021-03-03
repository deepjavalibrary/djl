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

import ai.djl.ndarray.NDManager;
import ai.djl.repository.zoo.ModelZoo;
import java.io.Serializable;

/**
 * Block factory is a component to make standard for block creating and saving procedure. Block
 * factory design is intended to bypass the serialization of the blocks. This class can be used by
 * {@link ModelZoo} or DJL Serving to recover the block to its uninitialized states. User should
 * combine this method with the block.loadParameter to get the block with all parameters.
 */
public interface BlockFactory extends Serializable {

    /**
     * Constructs the uninitialized block.
     *
     * @param manager the manager to assign to block
     * @return the uninitialized block
     */
    Block newBlock(NDManager manager);
}
