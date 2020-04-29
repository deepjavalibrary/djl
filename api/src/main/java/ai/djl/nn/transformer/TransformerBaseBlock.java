/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.nn.transformer;

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
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.function.Function;

/**
 * Utility base class for implementing custom blocks. Provides utilities to handle nested child
 * blocks, parameters and serialization of state.
 */
public abstract class TransformerBaseBlock extends AbstractBlock {

    /**
     * The model version of this block, used for checking if parameters are still valid during
     * parameter loading.
     */
    protected final int version;

    /**
     * All direct children of this Block. Keys are names of the blocks. Use the {@link
     * TransformerBaseBlock#addChildBlock(String, Block)} method to add children. All children in
     * this map are automagically loaded / saved.
     */
    // Using LinkedHashMap instead of Map is intentional: we want to make sure that consumers
    // of this API know the children are always iterated over in insertion order. LinkedHashMap
    // provides this guarantee, Map does not.
    @SuppressWarnings("PMD.LooseCoupling")
    protected final LinkedHashMap<String, Block> children = new LinkedHashMap<>();

    /**
     * All direct parameters of this Block. Keys are name of the parameters. Use the {@link
     * TransformerBaseBlock#addParameter(Parameter)} method to add children. All parameters in this
     * map are automagically loaded / saved.
     */
    // Using LinkedHashMap instead of Map is intentional: we want to make sure that consumers
    // of this API know the parameters are always iterated over in insertion order. LinkedHashMap
    // provides this guarantee, Map does not.
    @SuppressWarnings("PMD.LooseCoupling")
    protected final LinkedHashMap<String, Parameter> parameters = new LinkedHashMap<>();

    /**
     * Callbacks to determine the shape of a parameter. Values may be null in which case extending
     * classes need to override {@link Block#getParameterShape(String, Shape[])} and implement
     * parameter shape resolution manually.
     */
    // Using LinkedHashMap instead of Map is intentional: we want to make sure that consumers
    // of this API know the callbacks are always iterated over in insertion order. LinkedHashMap
    // provides this guarantee, Map does not.
    @SuppressWarnings("PMD.LooseCoupling")
    protected final LinkedHashMap<String, Function<Shape[], Shape>> parameterShapeCallbacks =
            new LinkedHashMap<>();

    /**
     * Builds an empty block with the given version for parameter serialization.
     *
     * @param version the version to use for parameter serialization.
     */
    public TransformerBaseBlock(final int version) {
        this.version = version;
    }

    /**
     * Returns the version number to be used for parameter serialization.
     *
     * @return the version number to be used for parameter serialization
     */
    public int getVersion() {
        return version;
    }

    /**
     * Adds a child block to this block.
     *
     * @param name Name of the block, must be unique or otherwise existing children with this name
     *     are removed, must not be null.
     * @param block The block, must not be null.
     * @param <B> The type of block
     * @return the block given as a parameter - that way the block can be created and reassigned to
     *     a member variable more easily.
     */
    protected <B extends Block> B addChildBlock(final String name, final B block) {
        children.put(name, block);
        return block;
    }

    /**
     * Adds a parameter to this block. If parameters are added with this method, subclasses need to
     * override {@link Block#getParameterShape(String, Shape[])} and return the shapes of parameters
     * themselves.
     *
     * @param parameter the parameter to add, not null
     * @param <P> the specific parameter subclass
     * @return the parameter passed as arguments to make it easier to create and assign paramters in
     *     one line
     */
    protected <P extends Parameter> P addParameter(final P parameter) {
        return addParameter(parameter, (Function<Shape[], Shape>) null);
    }

    /**
     * Adds a parameter to this block. If parameters are added with this method, intialization of
     * the parameter works out of the box
     *
     * @param parameter the parameter to add, not null
     * @param shape the shape of the parameter
     * @param <P> the specific parameter subclass
     * @return the parameter passed as arguments to make it easier to create and assign paramters in
     *     one line
     */
    protected <P extends Parameter> P addParameter(final P parameter, final Shape shape) {
        return addParameter(parameter, (inputShapes) -> shape);
    }

    /**
     * Adds a parameter to this block. If parameters are added with this method, intialization of
     * the parameter works out of the box
     *
     * @param parameter the parameter to add, not null
     * @param shapeCallback the method to call once the input shape of this block is known to
     *     determine the shape of the given parameter
     * @param <P> the specific parameter subclass
     * @return the parameter passed as arguments to make it easier to create and assign parameters
     *     in one line
     */
    protected <P extends Parameter> P addParameter(
            final P parameter, final Function<Shape[], Shape> shapeCallback) {
        parameters.put(parameter.getName(), parameter);
        parameterShapeCallbacks.put(parameter.getName(), shapeCallback);
        return parameter;
    }

    @Override
    public Shape getParameterShape(final String name, final Shape[] inputShapes) {
        final Function<Shape[], Shape> callback = parameterShapeCallbacks.get(name);
        if (callback == null) {
            final Parameter parameter = parameters.get(name);
            if (parameter == null) {
                throw new IllegalArgumentException(
                        "No parameter named " + name + " found in this block.");
            } else {
                throw new IllegalStateException(
                        "No shape initializer for parameter "
                                + name
                                + "found. "
                                + "Either pass an initializer for the shape when adding the parameter or override "
                                + "getParameterShape in the subclass.");
            }
        }
        return callback.apply(inputShapes);
    }

    @Override
    public BlockList getChildren() {
        return new BlockList(children);
    }

    /** {@inheritDoc} */
    @Override
    public final Shape[] initialize(
            final NDManager manager, final DataType dataType, final Shape... inputShapes) {
        beforeInitialize(inputShapes);
        for (final Parameter parameter : getDirectParameters()) {
            parameter.initialize(manager, dataType, inputShapes);
        }
        initializeChildBlocks(manager, dataType, inputShapes);
        return getOutputShapes(manager, inputShapes);
    }

    /**
     * Initializes the Child blocks of this block.
     *
     * @param manager the manager to use for initialization
     * @param dataType the requested data type
     * @param inputShapes the expected input shapes
     */
    public abstract void initializeChildBlocks(
            final NDManager manager, final DataType dataType, final Shape... inputShapes);

    /**
     * Default implementation of predict that calls forward.
     *
     * @param parameterStore the parameter store
     * @param inputs the input NDList
     * @param params optional parameters
     * @return the prediction result, same as from forward pass
     */
    @Override
    public NDList predict(
            ParameterStore parameterStore, NDList inputs, PairList<String, Object> params) {
        return forward(parameterStore, inputs, params);
    }

    @Override
    public List<Parameter> getDirectParameters() {
        return new ArrayList<>(parameters.values());
    }

    @Override
    public void saveParameters(final DataOutputStream os) throws IOException {
        os.write(version);
        for (final Parameter parameter : parameters.values()) {
            parameter.save(os);
        }
        for (final Block child : children.values()) {
            child.saveParameters(os);
        }
    }

    @Override
    public void loadParameters(final NDManager manager, final DataInputStream is)
            throws IOException, MalformedModelException {
        final int loadVersion = is.readInt();
        if (loadVersion != getVersion()) {
            throw new MalformedModelException(
                    "Cannot load parameters for "
                            + this.getClass().getCanonicalName()
                            + ", expected version "
                            + getVersion()
                            + ", got "
                            + loadVersion
                            + ".");
        }
        for (final Parameter parameter : parameters.values()) {
            parameter.load(manager, is);
        }
        for (final Block child : children.values()) {
            child.loadParameters(manager, is);
        }
    }
}
