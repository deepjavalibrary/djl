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
package software.amazon.ai.nn;

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
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.ndarray.NDManager;
import software.amazon.ai.ndarray.types.Shape;
import software.amazon.ai.util.PairList;

/** ParallelBlock can be used to represent branches in the computational graph. */
public class ParallelBlock extends AbstractBlock {

    private static final byte VERSION = 1;

    private List<Block> blocks;
    private Function<List<NDList>, NDList> function;

    public ParallelBlock(Function<List<NDList>, NDList> function) {
        this.function = function;
        blocks = new ArrayList<>();
    }

    public ParallelBlock(Function<List<NDList>, NDList> function, List<Block> blocks) {
        this.function = function;
        this.blocks = blocks;
    }

    public ParallelBlock addAll(Block... blocks) {
        this.blocks.addAll(Arrays.asList(blocks));
        initialized = false;
        return this;
    }

    public ParallelBlock addAll(Collection<Block> blocks) {
        this.blocks.addAll(blocks);
        initialized = false;
        return this;
    }

    public ParallelBlock add(Block block) {
        blocks.add(block);
        initialized = false;
        return this;
    }

    public ParallelBlock add(Function<NDList, NDList> f) {
        blocks.add(new LambdaBlock(f));
        initialized = false;
        return this;
    }

    @Override
    public NDList forward(NDList inputs, PairList<String, Object> params) {
        return function.apply(
                blocks.stream()
                        .map(
                                block -> {
                                    block.initialize(inputs);
                                    return block.forward(inputs, params);
                                })
                        .collect(Collectors.toList()));
    }

    @Override
    public Shape getOutputShape(Shape... inputs) {
        return null;
    }

    @Override
    public List<Parameter> getDirectParameters() {
        return Collections.emptyList();
    }

    @Override
    public void beforeInitialize(NDList inputs) {
        initialized = true;
    }

    @Override
    public Shape getParameterShape(String name, Shape[] inputShapes) {
        throw new IllegalArgumentException("ParallelBlock have no parameters");
    }

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

    @Override
    public void saveParameters(DataOutputStream os) throws IOException {
        os.writeByte(VERSION);
        for (Block block : blocks) {
            block.saveParameters(os);
        }
    }

    @Override
    public void loadParameters(NDManager manager, DataInputStream is) throws IOException {
        byte version = is.readByte();
        if (version != VERSION) {
            throw new IllegalArgumentException("Unsupported encoding version: " + version);
        }
        for (Block block : blocks) {
            block.loadParameters(manager, is);
        }
    }
}
