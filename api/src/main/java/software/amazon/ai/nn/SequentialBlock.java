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
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.ndarray.NDManager;
import software.amazon.ai.ndarray.types.DataDesc;
import software.amazon.ai.ndarray.types.DataType;
import software.amazon.ai.ndarray.types.Shape;
import software.amazon.ai.util.PairList;

public class SequentialBlock extends AbstractBlock {

    private static final byte VERSION = 1;

    private List<Block> blocks = new ArrayList<>();

    public SequentialBlock addAll(Block... blocks) {
        this.blocks.addAll(Arrays.asList(blocks));
        initialized = false;
        return this;
    }

    public SequentialBlock addAll(Collection<Block> blocks) {
        this.blocks.addAll(blocks);
        initialized = false;
        return this;
    }

    public SequentialBlock add(Block block) {
        blocks.add(block);
        initialized = false;
        return this;
    }

    public SequentialBlock add(Function<NDList, NDList> f) {
        blocks.add(new LambdaBlock(f));
        initialized = false;
        return this;
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
        initialized = true;
    }

    @Override
    public void cast(DataType dataType) {
        blocks.forEach(ele -> ele.cast(dataType));
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
