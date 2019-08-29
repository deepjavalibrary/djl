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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.function.Function;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.ndarray.types.DataDesc;
import software.amazon.ai.ndarray.types.DataType;
import software.amazon.ai.ndarray.types.Shape;
import software.amazon.ai.util.PairList;

public class SequentialBlock implements Block {

    boolean isInitialized;
    List<Block> blocks;

    public SequentialBlock(Block... blocks) {
        this.blocks = new ArrayList<>(Arrays.asList(blocks));
    }

    public void addAll(Block... blocks) {
        this.blocks.addAll(Arrays.asList(blocks));
        this.isInitialized = false;
    }

    public void addAll(Collection<Block> blocks) {
        this.blocks.addAll(blocks);
        this.isInitialized = false;
    }

    public void add(Block block) {
        blocks.add(block);
        this.isInitialized = false;
    }

    public void add(Function<NDList, NDList> f) {
        blocks.add(new LambdaBlock(f));
        this.isInitialized = false;
    }

    public void removeLastBlock() {
        blocks.remove(blocks.size() - 1);
    }

    public void replaceLastBlock(Block block) {
        removeLastBlock();
        blocks.add(block);
    }

    @Override
    public NDList forward(NDList inputs, PairList<String, Object> params) {
        NDList current = inputs;
        for (Block block : blocks) {
            NDList previous = current;
            block.ensureInitialized(current);
            current = block.forward(current);
            if (previous != inputs) {
                previous.close();
            }
        }
        return current;
    }

    @Override
    public boolean isInitialized() {
        return isInitialized;
    }

    @Override
    public Shape getOutputShape(Shape... inputs) {
        if (blocks.isEmpty()) {
            throw new IllegalArgumentException("The sequential block is empty");
        }
        Shape[] current = inputs;
        for (Block block : blocks) {
            current = new Shape[] {block.getOutputShape(current)};
        }
        return current[0];
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
    public Block cast(DataType dataType) {
        SequentialBlock newBlock = new SequentialBlock();
        blocks.forEach(ele -> newBlock.add(ele.cast(dataType)));
        return newBlock;
    }

    @Override
    public DataDesc[] describeInput() {
        if (blocks.isEmpty()) {
            throw new IllegalArgumentException("The sequential block is empty");
        }
        return blocks.get(0).describeInput();
    }

    @Override
    public Shape getParameterShape(String name, NDList inputs) {
        throw new IllegalArgumentException("SequentialBlocks have no parameters");
    }

    @Override
    public byte[] getEncoded() {
        return new byte[0];
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
}
