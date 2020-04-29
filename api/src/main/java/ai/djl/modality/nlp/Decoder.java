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
import ai.djl.nn.BlockList;
import ai.djl.nn.Parameter;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Collections;
import java.util.List;

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
     */
    public Decoder(Block block) {
        this.block = block;
    }

    /**
     * Sets the state of the encoder as the initial state of the decoder.
     *
     * @param encoderStates the states of the encoder
     */
    public abstract void initState(NDList encoderStates);

    /** {@inheritDoc} */
    @Override
    public NDList forward(
            ParameterStore parameterStore, NDList inputs, PairList<String, Object> params) {
        return block.forward(parameterStore, inputs, params);
    }

    /** {@inheritDoc} */
    @Override
    public Shape[] initialize(NDManager manager, DataType dataType, Shape... inputShapes) {
        beforeInitialize(inputShapes);
        return block.initialize(manager, dataType, inputShapes);
    }

    /** {@inheritDoc} */
    @Override
    public BlockList getChildren() {
        return new BlockList(Collections.singletonList("Block"), Collections.singletonList(block));
    }

    /** {@inheritDoc} */
    @Override
    public List<Parameter> getDirectParameters() {
        return Collections.emptyList();
    }

    /** {@inheritDoc} */
    @Override
    public Shape getParameterShape(String name, Shape[] inputShapes) {
        throw new IllegalArgumentException("Decoder has no parameters");
    }

    /** {@inheritDoc} */
    @Override
    public Shape[] getOutputShapes(NDManager manager, Shape[] inputShapes) {
        return block.getOutputShapes(manager, inputShapes);
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
