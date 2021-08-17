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
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import java.io.DataInputStream;
import java.io.IOException;
import java.util.Arrays;
import java.util.Collection;
import java.util.function.Function;

/**
 * {@code SequentialBlock} is a {@link Block} whose children form a chain of blocks with each child
 * block feeding its output to the next. The output of the last child is returned as the output of
 * the {@code SequentialBlock}.
 *
 * <p>{@code SequentialBlock} has no direct parameters.
 */
public class SequentialBlock extends AbstractBlock {

    private static final byte VERSION = 2;

    /**
     * Creates an empty sequential block. Use {@code add} and {@code addAll} to add blocks to be
     * executed in sequence.
     */
    public SequentialBlock() {
        super(VERSION);
    }

    /**
     * Adds an array of blocks to be executed in sequence, in order.
     *
     * @param blocks the array of blocks
     * @return this block
     */
    public SequentialBlock addAll(Block... blocks) {
        this.addAll(Arrays.asList(blocks));
        return this;
    }

    /**
     * Adds a {@link Collection} of blocks to be executed in sequence, in order.
     *
     * @param blocks the {@link Collection} of blocks
     * @return this block
     */
    public SequentialBlock addAll(Collection<Block> blocks) {
        blocks.forEach(this::add);
        return this;
    }

    /**
     * Adds the given {@link Block} to the block to be executed in order.
     *
     * @param block the block to be added to the sequence of blocks
     * @return this block
     */
    public SequentialBlock add(Block block) {
        if (block != null) {
            addChildBlock(block.getClass().getSimpleName(), block);
        }
        return this;
    }

    /**
     * Adds a {@link LambdaBlock} that applies the given function to the sequence of blocks.
     *
     * @param f the function forms the {@link LambdaBlock}
     * @return this block
     */
    public SequentialBlock add(Function<NDList, NDList> f) {
        add(new LambdaBlock(f));
        return this;
    }

    /**
     * Adds a {@link LambdaBlock#singleton(Function)} that applies the given function to the
     * sequence of blocks.
     *
     * @param f the function forms the {@link LambdaBlock}
     * @return this block
     * @see LambdaBlock#singleton(Function)
     */
    public SequentialBlock addSingleton(Function<NDArray, NDArray> f) {
        add(LambdaBlock.singleton(f));
        return this;
    }

    /** Removes the {@link Block} added last from the sequence of blocks. */
    public void removeLastBlock() {
        children.remove(children.size() - 1);
    }

    /**
     * Replaces the {@link Block} last added from the sequence of blocks, and adds the given block.
     *
     * @param block the block to replace the last block with
     */
    public void replaceLastBlock(Block block) {
        removeLastBlock();
        if (block != null) {
            add(block);
        }
    }

    /** {@inheritDoc} */
    @Override
    protected NDList forwardInternal(
            ParameterStore parameterStore,
            NDList inputs,
            boolean training,
            PairList<String, Object> params) {
        NDList current = inputs;
        for (Block block : children.values()) {
            current = block.forward(parameterStore, current, training);
        }
        return current;
    }

    /** {@inheritDoc} */
    @Override
    protected NDList forwardInternal(
            ParameterStore parameterStore,
            NDList data,
            NDList labels,
            PairList<String, Object> params) {
        NDList current = data;
        for (Block block : children.values()) {
            current = block.forward(parameterStore, current, labels, params);
        }
        return current;
    }

    /** {@inheritDoc} */
    @Override
    public void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {
        Shape[] shapes = inputShapes;
        for (Block child : getChildren().values()) {
            child.initialize(manager, dataType, shapes);
            shapes = child.getOutputShapes(shapes);
        }
    }

    /** {@inheritDoc} */
    @Override
    public Shape[] getOutputShapes(Shape[] inputs) {
        if (children.isEmpty()) {
            throw new IllegalArgumentException("The sequential block is empty");
        }
        Shape[] current = inputs;
        for (Block block : children.values()) {
            current = block.getOutputShapes(current);
        }
        return current;
    }

    /** {@inheritDoc} */
    @Override
    public void loadMetadata(byte loadVersion, DataInputStream is)
            throws IOException, MalformedModelException {
        if (loadVersion == version) {
            readInputShapes(is);
        } else if (loadVersion != 1) {
            throw new MalformedModelException("Unsupported encoding version: " + loadVersion);
        }
    }

    /** {@inheritDoc} */
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder(200);
        sb.append("Sequential(\n");
        for (Block block : children.values()) {
            String blockString = block.toString().replaceAll("(?m)^", "\t");
            sb.append(blockString).append('\n');
        }
        sb.append(')');
        return sb.toString();
    }
}
