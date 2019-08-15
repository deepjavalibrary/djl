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
package software.amazon.ai;

import java.util.Collections;
import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.ndarray.types.DataDesc;
import software.amazon.ai.ndarray.types.Shape;
import software.amazon.ai.util.PairList;

/** ParallelBlock can be used to represent branches in the computational graph. */
public class ParallelBlock implements Block {
    private List<Block> blocks;
    private Function<List<NDList>, NDList> function;
    private boolean isInitialized;

    public ParallelBlock(List<Block> blocks, Function<List<NDList>, NDList> function) {
        this.blocks = blocks;
        this.function = function;
    }

    @Override
    public NDList forward(NDList inputs, PairList<String, Object> params) {
        return function.apply(
                blocks.stream()
                        .map(
                                block -> {
                                    block.ensureInitialized(inputs);
                                    return block.forward(inputs, params);
                                })
                        .collect(Collectors.toList()));
    }

    @Override
    public boolean isInitialized() {
        return isInitialized;
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
        isInitialized = true;
    }

    @Override
    public DataDesc[] describeInput() {
        return new DataDesc[0];
    }

    @Override
    public Shape getParameterShape(String name, NDList inputs) {
        throw new IllegalArgumentException("ParallelBlock have no parameters");
    }

    @Override
    public PairList<String, Block> getChildren() {
        PairList<String, Block> children = new PairList<>(blocks.size());
        for (int i = 0; i < blocks.size(); i++) {
            Block block = blocks.get(i);
            String name = String.format("%02d:%s", i, block.getClass().getSimpleName());
            children.add(name, block);
        }
        return children;
    }

    @Override
    public byte[] getEncoded() {
        return new byte[0];
    }
}
