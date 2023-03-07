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

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.ParameterStore;
import ai.djl.util.Pair;
import ai.djl.util.PairList;

import java.io.DataOutputStream;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.function.Function;

/**
 * {@code AbstractBlock} is an abstract implementation of {@link Block}.
 *
 * <p>It is recommended that all {@code Block} classes that have children extend the {@code
 * AbstractBlock}.
 *
 * <p>To create your own blocks, you need to do the following:
 *
 * <ul>
 *   <li>Define a version for serializing parameter and metadata and pass it to the parent
 *       constructor
 *   <li>Use {@link AbstractBlock#addParameter(Parameter)} to add parameters to your block in the
 *       constructor if necessary.
 *   <li>Use {@link AbstractBlock#addChildBlock(String, Block)} to add child blocks if necessary.
 *   <li>Override {@link Block#getOutputShapes(Shape[])} to determine the shape of your custom
 *       block's output based on the input it will receive.
 *   <li>Override {@link AbstractBlock#initializeChildBlocks(NDManager, DataType, Shape...)} if you
 *       added child blocks to initialize them based on the input shape your block will receive. You
 *       can skip this if your block does not contain child blocks
 *   <li>Override {@link AbstractBlock#forward(ParameterStore, NDList, boolean, PairList)} to
 *       implement the computation of your block
 *   <li>IFF you need to save data apart from the parameter values of your block, you need to
 *       override {@link AbstractBlock#saveMetadata(DataOutputStream)} and {@link
 *       AbstractBlock#loadMetadata(byte, java.io.DataInputStream)}. If you do not need to save or
 *       load any state other than parameters in your block, you can skip this.
 * </ul>
 *
 * <p>If you use {@link AbstractBlock#addParameter(Parameter)} to add parameters, you have to take
 * care of parameter initialization yourself. In this case, you need to setShape to your parameters
 * if you know the shape of Parameter or you can implement prepare to setShape when you see the
 * input shape.
 */
// Using LinkedHashMap instead of Map is intentional: we want to make sure that consumers
// of this API know the children and parameters are always iterated over in insertion order.
// LinkedHashMap provides this guarantee, Map does not.
@SuppressWarnings("PMD.LooseCoupling")
public abstract class AbstractBlock extends AbstractBaseBlock {

    /**
     * All direct children of this Block. Keys are names of the blocks.
     *
     * <p>Use the {@link AbstractBlock#addChildBlock(String, Block)} method to add children. All
     * children in this map are automagically loaded / saved.
     */
    protected BlockList children = new BlockList();

    /**
     * All direct parameters of this Block. Keys are name of the parameters.
     *
     * <p>Use the {@link AbstractBlock#addParameter(Parameter)} method to add children. All
     * parameters in this map are automatically loaded / saved.
     */
    protected LinkedHashMap<String, Parameter> parameters = new LinkedHashMap<>();

    /** Constructs a new {@code AbstractBlock} instance. */
    public AbstractBlock() {}

    /**
     * Builds an empty block with the given version for parameter serialization.
     *
     * @param version the version to use for parameter serialization.
     */
    public AbstractBlock(byte version) {
        super(version);
    }

    /**
     * Constructs a copy of another {@link AbstractBlock}.
     *
     * @param block the block to copy
     */
    public AbstractBlock(AbstractBlock block) {
        super(block);
        children = block.children;
        parameters = block.parameters;
    }

    /**
     * Use this to add a child block to this block.
     *
     * @param name Name of the block, must not be null.
     * @param block The block, must not be null.
     * @param <B> The type of block
     * @return the block given as a parameter - that way the block can be created and reassigned to
     *     a member variable more easily.
     */
    protected final <B extends Block> B addChildBlock(String name, B block) {
        int childNumber = children.size() + 1;
        children.add(String.format(Locale.ROOT, "%02d%s", childNumber, name), block);
        return block;
    }

    /**
     * Adds a {@link LambdaBlock} as a child block to this block.
     *
     * @param name Name of the block, must not be null.
     * @param f the function forms the {@link LambdaBlock}
     * @return the child block
     */
    protected LambdaBlock addChildBlock(String name, Function<NDList, NDList> f) {
        return addChildBlock(name, new LambdaBlock(f, name));
    }

    /**
     * Adds a {@link LambdaBlock#singleton(Function)} as a child block to this block.
     *
     * @param name Name of the block, must not be null.
     * @param f the function forms the {@link LambdaBlock}
     * @return the child block
     * @see LambdaBlock#singleton(Function)
     */
    protected final LambdaBlock addChildBlockSingleton(String name, Function<NDArray, NDArray> f) {
        return addChildBlock(name, LambdaBlock.singleton(f, name));
    }

    /**
     * Adds a parameter to this block. If parameters are added with this method, initialization of
     * the parameter works out of the box
     *
     * @param <P> the specific parameter subclass
     * @param parameter the parameter to add, not null
     * @return the parameter passed as arguments to make it easier to create and assign parameters
     *     in one line
     */
    protected final <P extends Parameter> P addParameter(P parameter) {
        parameters.put(parameter.getName(), parameter);
        return parameter;
    }

    /** {@inheritDoc} */
    @Override
    public BlockList getChildren() {
        BlockList defensiveCopy = new BlockList(children.size());
        for (Pair<String, Block> entry : children) {
            defensiveCopy.add(entry);
        }
        return defensiveCopy;
    }

    /** {@inheritDoc} */
    @Override
    public ParameterList getDirectParameters() {
        return new ParameterList(parameters);
    }
}
