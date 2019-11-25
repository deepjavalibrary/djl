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

import ai.djl.MalformedModelException;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.function.Function;

/**
 * {@code SequentialBlock} is a {@link Block} whose children form a chain of blocks with each child
 * block feeding its output to the next. The output of the last child is returned as the output of
 * the {@code SequentialBlock}.
 *
 * <p>{@code SequentialBlock} has no direct parameters.
 */
public class SequentialBlock extends AbstractBlock {

    private static final byte VERSION = 1;

    private List<Block> blocks = new ArrayList<>();

    /**
     * Adds an array of blocks to be executed in sequence, in order.
     *
     * @param blocks the array of blocks
     * @return this block
     */
    public SequentialBlock addAll(Block... blocks) {
        this.blocks.addAll(Arrays.asList(blocks));
        return this;
    }

    /**
     * Adds a {@link Collection} of blocks to be executed in sequence, in order.
     *
     * @param blocks the {@link Collection} of blocks
     * @return this block
     */
    public SequentialBlock addAll(Collection<Block> blocks) {
        this.blocks.addAll(blocks);
        return this;
    }

    /**
     * Adds the given {@link Block} to the block to be executed in order.
     *
     * @param block the block to be added to the sequence of blocks
     * @return this block
     */
    public SequentialBlock add(Block block) {
        blocks.add(block);
        return this;
    }

    /**
     * Adds a {@link LambdaBlock} that applies the given function to the sequence of blocks.
     *
     * @param f the function forms the {@link LambdaBlock}
     * @return this block
     */
    public SequentialBlock add(Function<NDList, NDList> f) {
        blocks.add(new LambdaBlock(f));
        return this;
    }

    /** Removes the {@link Block} added last from the sequence of blocks. */
    public void removeLastBlock() {
        blocks.remove(blocks.size() - 1);
    }

    /**
     * Replaces the {@link Block} last added from the sequence of blocks, and adds the given block.
     *
     * @param block the block to replace the last block with
     */
    public void replaceLastBlock(Block block) {
        removeLastBlock();
        blocks.add(block);
    }

    /** {@inheritDoc} */
    @Override
    public NDList forward(
            ParameterStore parameterStore, NDList inputs, PairList<String, Object> params) {
        NDList current = inputs;
        for (Block block : blocks) {
            current = block.forward(parameterStore, current);
        }
        return current;
    }

    /** {@inheritDoc} */
    @Override
    public Shape[] initialize(NDManager manager, DataType dataType, Shape... inputShapes) {
        beforeInitialize(inputShapes);
        Shape[] shapes = inputShapes;
        for (Block child : getChildren().values()) {
            shapes = child.initialize(manager, dataType, shapes);
        }
        return getOutputShapes(manager, inputShapes);
    }

    /** {@inheritDoc} */
    @Override
    public Shape[] getOutputShapes(NDManager manager, Shape[] inputs) {
        if (blocks.isEmpty()) {
            throw new IllegalArgumentException("The sequential block is empty");
        }
        Shape[] current = inputs;
        for (Block block : blocks) {
            current = block.getOutputShapes(manager, current);
        }
        return current;
    }

    /** {@inheritDoc} */
    @Override
    public List<Parameter> getDirectParameters() {
        return Collections.emptyList();
    }

    /** {@inheritDoc} */
    @Override
    public Shape getParameterShape(String name, Shape[] inputShapes) {
        throw new IllegalArgumentException("SequentialBlocks have no parameters");
    }

    /** {@inheritDoc} */
    @Override
    public BlockList getChildren() {
        int size = blocks.size();
        BlockList children = new BlockList(size);
        int precision = (int) Math.log10(size) + 1;
        String format = "%0" + precision + "d:%s";
        for (int i = 0; i < size; ++i) {
            Block block = blocks.get(i);
            String name = String.format(format, i, block.getClass().getSimpleName());
            children.add(name, block);
        }
        return children;
    }

    /** {@inheritDoc} */
    @Override
    public void saveParameters(DataOutputStream os) throws IOException {
        os.writeByte(VERSION);
        for (Block block : blocks) {
            block.saveParameters(os);
        }
    }

    /** {@inheritDoc} */
    @Override
    public void loadParameters(NDManager manager, DataInputStream is)
            throws IOException, MalformedModelException {
        byte version = is.readByte();
        if (version != VERSION) {
            throw new MalformedModelException("Unsupported encoding version: " + version);
        }
        for (Block block : blocks) {
            block.loadParameters(manager, is);
        }
    }

    /** {@inheritDoc} */
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder(200);
        sb.append("Sequential(\n");
        for (Block block : blocks) {
            sb.append('\t').append(block).append('\n');
        }
        sb.append(')');
        return sb.toString();
    }
}
