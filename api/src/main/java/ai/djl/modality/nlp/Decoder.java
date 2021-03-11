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
package ai.djl.modality.nlp;

import ai.djl.MalformedModelException;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.Block;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

/**
 * {@code Decoder} is an abstract block that be can used as decoder in encoder-decoder architecture.
 * This abstraction, along with {@link Encoder}, comes into play in the {@link EncoderDecoder}
 * class, and facilitate implementing encoder-decoder models for different tasks and inputs.
 */
public abstract class Decoder extends AbstractBlock {

    protected Block block;

    /**
     * Constructs a new instance of {@code Decoder} with the given block. Use this constructor if
     * you are planning to use pre-trained embeddings that don't need further training.
     *
     * @param block the block to be used to decode
     * @param version the version to use for parameter and metadata serialization
     */
    public Decoder(byte version, Block block) {
        super(version);
        this.block = addChildBlock("Block", block);
    }

    /** {@inheritDoc} */
    @Override
    protected NDList forwardInternal(
            ParameterStore parameterStore,
            NDList inputs,
            boolean training,
            PairList<String, Object> params) {
        return block.forward(parameterStore, inputs, training, params);
    }

    /** {@inheritDoc} */
    @Override
    public void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {
        block.initialize(manager, dataType, inputShapes);
    }

    /** {@inheritDoc} */
    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        return block.getOutputShapes(inputShapes);
    }

    /** {@inheritDoc} */
    @Override
    public void saveParameters(DataOutputStream os) throws IOException {
        block.saveParameters(os);
    }

    /** {@inheritDoc} */
    @Override
    public void loadParameters(NDManager manager, DataInputStream is)
            throws IOException, MalformedModelException {
        block.loadParameters(manager, is);
    }
}
