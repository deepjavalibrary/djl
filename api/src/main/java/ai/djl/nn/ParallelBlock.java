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
import java.util.stream.Collectors;

/**
 * {@code ParallelBlock} is a {@link Block} whose children form a parallel branch in the network and
 * are combined to produce a single output.
 *
 * <p>{@code ParallelBlock} has no direct parameters.
 */
public class ParallelBlock extends AbstractBlock {

    private static final byte VERSION = 1;

    private List<Block> blocks;
    private Function<List<NDList>, NDList> function;

    /**
     * Creates a parallel block whose branches are combined to form a single output by the given
     * function.
     *
     * @param function the function to define how the parallel branches are combined to form a
     *     single output
     */
    public ParallelBlock(Function<List<NDList>, NDList> function) {
        this.function = function;
        blocks = new ArrayList<>();
    }

    /**
     * Creates a parallel block whose branches are formed by each block in the list of blocks, and
     * are combined to form a single output by the given function.
     *
     * @param function the function to define how the parallel branches are combined
     * @param blocks the blocks that form each of the parallel branches
     */
    public ParallelBlock(Function<List<NDList>, NDList> function, List<Block> blocks) {
        this.function = function;
        this.blocks = blocks;
    }

    /**
     * Adds an array of blocks, each of which is a parallel branch.
     *
     * @param blocks the array of blocks to add
     * @return this block
     */
    public final ParallelBlock addAll(Block... blocks) {
        this.blocks.addAll(Arrays.asList(blocks));
        return this;
    }

    /**
     * Adds a {@link Collection} of blocks, each of which is a parallel branch.
     *
     * @param blocks the {@link Collection} of blocks to add
     * @return this block
     */
    public final ParallelBlock addAll(Collection<Block> blocks) {
        this.blocks.addAll(blocks);
        return this;
    }

    /**
     * Adds the given {@link Block} to the block, which is one parallel branch.
     *
     * @param block the block to be added as a parallel branch
     * @return this block
     */
    public final ParallelBlock add(Block block) {
        blocks.add(block);
        return this;
    }

    /**
     * Adds a {@link LambdaBlock}, that applies the given function, to the list of parallel
     * branches.
     *
     * @param f the function that forms the {@link LambdaBlock}
     * @return this block
     */
    public final ParallelBlock add(Function<NDList, NDList> f) {
        blocks.add(new LambdaBlock(f));
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public NDList forward(
            ParameterStore parameterStore, NDList inputs, PairList<String, Object> params) {
        return function.apply(
                blocks.stream()
                        .map(block -> block.forward(parameterStore, inputs, params))
                        .collect(Collectors.toList()));
    }

    /** {@inheritDoc} */
    @Override
    public Shape[] initialize(NDManager manager, DataType dataType, Shape... inputShapes) {
        beforeInitialize(inputShapes);
        for (Block child : getChildren().values()) {
            child.initialize(manager, dataType, inputShapes);
        }
        return getOutputShapes(manager, inputShapes);
    }

    /** {@inheritDoc} */
    @Override
    public Shape[] getOutputShapes(NDManager manager, Shape[] inputShapes) {
        if (blocks.isEmpty()) {
            throw new IllegalArgumentException("The sequential block is empty");
        }

        try (NDManager subManager = manager.newSubManager()) {
            List<NDList> inputs = new ArrayList<>();
            for (Block block : blocks) {
                Shape[] shapes = block.getOutputShapes(manager, inputShapes);
                NDList output = new NDList(shapes.length);
                for (Shape shape : shapes) {
                    output.add(subManager.create(shape));
                }
                inputs.add(output);
            }
            NDList output = function.apply(inputs);
            Shape[] outputShapes = new Shape[output.size()];
            for (int i = 0; i < output.size(); ++i) {
                outputShapes[i] = output.get(i).getShape();
            }
            return outputShapes;
        }
    }

    /** {@inheritDoc} */
    @Override
    public List<Parameter> getDirectParameters() {
        return Collections.emptyList();
    }

    /** {@inheritDoc} */
    @Override
    public Shape getParameterShape(String name, Shape[] inputShapes) {
        throw new IllegalArgumentException("ParallelBlock have no parameters");
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
        sb.append("Parallel(\n");
        for (Block block : blocks) {
            sb.append('\t').append(block).append('\n');
        }
        sb.append(')');
        return sb.toString();
    }
}
