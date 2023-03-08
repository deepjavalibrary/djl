/*
 * Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import ai.djl.training.initializer.Initializer;
import ai.djl.util.Pair;
import ai.djl.util.PairList;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.function.Predicate;

/**
 * This provides shared functionality for both the DJL-based {@link AbstractBlock}s and the imported
 * {@link AbstractSymbolBlock}s.
 */
public abstract class AbstractBaseBlock implements Block {

    /**
     * The model version of this block, used for checking if parameters are still valid during
     * parameter loading.
     */
    protected byte version;

    /** The shape of the input for this block, set by the initialization process. */
    protected Shape[] inputShapes;

    protected DataType[] outputDataTypes;

    /** List of names for the input, named inputs should be manually set in sub class. */
    protected List<String> inputNames = Collections.emptyList();

    /** Constructs a new {@link AbstractBaseBlock} instance. */
    public AbstractBaseBlock() {
        this((byte) 1);
    }

    /**
     * Builds an empty block with the given version for parameter serialization.
     *
     * @param version the version to use for parameter serialization.
     */
    public AbstractBaseBlock(byte version) {
        this.version = version;
    }

    /** {@inheritDoc} */
    @Override
    public final NDList forward(
            ParameterStore parameterStore,
            NDList inputs,
            boolean training,
            PairList<String, Object> params) {
        NDManager paramsManager = parameterStore.getManager();
        if (training && !isInitialized()) {
            initialize(paramsManager, DataType.FLOAT32, inputs.getShapes());
        }
        return forwardInternal(parameterStore, inputs, training, params);
    }

    /** {@inheritDoc} */
    @Override
    public NDList forward(
            ParameterStore parameterStore,
            NDList data,
            NDList labels,
            PairList<String, Object> params) {
        NDManager paramsManager = parameterStore.getManager();
        if (!isInitialized()) {
            initialize(paramsManager, DataType.FLOAT32, data.getShapes());
        }
        return forwardInternal(parameterStore, data, labels, params);
    }

    /**
     * A helper for {@link Block#forward(ParameterStore, NDList, boolean, PairList)} after
     * initialization.
     *
     * @param parameterStore the parameter store
     * @param inputs the input NDList
     * @param training true for a training forward pass
     * @param params optional parameters
     * @return the output of the forward pass
     */
    protected abstract NDList forwardInternal(
            ParameterStore parameterStore,
            NDList inputs,
            boolean training,
            PairList<String, Object> params);

    /**
     * A helper for {@link Block#forward(ParameterStore, NDList, NDList, PairList)} after
     * initialization.
     *
     * @param parameterStore the parameter store
     * @param data the input data NDList
     * @param labels the input labels NDList
     * @param params optional parameters
     * @return the output of the forward pass
     * @see #forward(ParameterStore, NDList, boolean, PairList)
     */
    protected NDList forwardInternal(
            ParameterStore parameterStore,
            NDList data,
            NDList labels,
            PairList<String, Object> params) {
        return forwardInternal(parameterStore, data, true, params);
    }

    /** {@inheritDoc} */
    @Override
    public PairList<String, Shape> describeInput() {
        if (!isInitialized()) {
            throw new IllegalStateException(
                    "Parameter of this block are not initialised,"
                            + "please call model.newTrainer and trainer.initialize");
        }
        return new PairList<>(inputNames, Arrays.asList(inputShapes));
    }

    /** {@inheritDoc} */
    @Override
    public void setInitializer(Initializer initializer, Parameter.Type params) {
        Predicate<Parameter> predicate = parameter -> parameter.getType().equals(params);
        setInitializer(initializer, predicate);
    }

    /** {@inheritDoc} */
    @Override
    public void setInitializer(Initializer initializer, String paramName) {
        Parameter parameter =
                getDirectParameters().values().stream()
                        .filter(p -> p.getName().equals(paramName))
                        .findFirst()
                        .orElseThrow(
                                () ->
                                        new IllegalArgumentException(
                                                "Could not find parameter " + paramName));
        parameter.setInitializer(initializer);
    }

    /** {@inheritDoc} */
    @Override
    public void setInitializer(Initializer initializer, Predicate<Parameter> predicate) {
        List<Parameter> params = getParameters().values();
        for (Parameter param : params) {
            if (predicate.test(param)) {
                param.setInitializer(initializer);
            }
        }
    }

    /** {@inheritDoc} */
    @Override
    public void initialize(NDManager manager, DataType dataType, Shape... inputShapes) {
        beforeInitialize(inputShapes);
        // Block inputShape is null or params arrays are null
        if (!isInitialized()) {
            // Set the shape of parameter to be inputShapes
            prepare(inputShapes);
        }
        for (Parameter parameter : getDirectParameters().values()) {
            // Attach arrays to params if params are null; set require gradient if required
            parameter.initialize(manager, dataType);
        }
        initializeChildBlocks(manager, dataType, inputShapes);
    }

    /**
     * Performs any action necessary before initialization. For example, keep the input information
     * or verify the layout.
     *
     * @param inputShapes the expected shapes of the input
     */
    protected void beforeInitialize(Shape... inputShapes) {
        if (inputNames.isEmpty()) {
            // automatically assign input names
            inputNames = new ArrayList<>();
            for (int i = 0; i < inputShapes.length; ++i) {
                inputNames.add("data" + i);
            }
        }
        this.inputShapes = inputShapes;
    }

    /**
     * Initializes the Child blocks of this block. You need to override this method if your subclass
     * has child blocks. Used to determine the correct input shapes for child blocks based on the
     * requested input shape for this block.
     *
     * @param manager the manager to use for initialization
     * @param dataType the requested data type
     * @param inputShapes the expected input shapes for this block
     */
    protected void initializeChildBlocks(
            NDManager manager, DataType dataType, Shape... inputShapes) {
        if (!getChildren().isEmpty()) {
            throw new IllegalStateException(
                    getClass().getSimpleName()
                            + " has child blocks but initializeChildBlocks is not overwritten.");
        }
    }

    /**
     * Sets the shape of {@link Parameter}s.
     *
     * @param inputShapes the shapes of inputs
     */
    protected void prepare(Shape[] inputShapes) {}

    /** {@inheritDoc} */
    @Override
    public ParameterList getParameters() {
        // we accumulate a list of all parameters by starting with a list of the direct parameters
        ParameterList allParams = getDirectParameters();
        // then we add the parameters of child blocks
        for (Pair<String, Block> childPair : getChildren()) {
            for (Pair<String, Parameter> paramPair : childPair.getValue().getParameters()) {
                // we prepend the name of the child block to the parameter name
                allParams.add(childPair.getKey() + "_" + paramPair.getKey(), paramPair.getValue());
            }
        }
        return allParams;
    }

    /** {@inheritDoc} */
    @Override
    public boolean isInitialized() {
        if (inputShapes == null) {
            return false;
        }
        for (Parameter param : getParameters().values()) {
            if (!param.isInitialized()) {
                return false;
            }
        }
        return true;
    }

    /** {@inheritDoc} */
    @Override
    public void clear() {
        getParameters().forEach(param -> param.getValue().close());
    }

    /** {@inheritDoc} */
    @Override
    public void cast(DataType dataType) {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    /** {@inheritDoc} */
    @Override
    public void saveParameters(DataOutputStream os) throws IOException {
        os.write(version);
        saveMetadata(os);
        for (Parameter parameter : getDirectParameters().values()) {
            parameter.save(os);
        }
        for (Block child : getChildren().values()) {
            child.saveParameters(os);
        }
    }

    /** {@inheritDoc} */
    @Override
    public void loadParameters(NDManager manager, DataInputStream is)
            throws IOException, MalformedModelException {
        byte loadVersion = is.readByte();
        loadMetadata(loadVersion, is);
        for (Parameter parameter : getDirectParameters().values()) {
            parameter.load(manager, is);
        }
        for (Block child : getChildren().values()) {
            child.loadParameters(manager, is);
        }
    }

    /**
     * Override this method to save additional data apart from parameter values.
     *
     * <p>This default implementation saves the currently set input shapes.
     *
     * @param os the non-null output stream the parameter values and metadata are written to
     * @throws IOException saving failed
     */
    protected void saveMetadata(DataOutputStream os) throws IOException {
        saveInputShapes(os);
    }

    /**
     * Overwrite this to load additional metadata with the parameter values.
     *
     * <p>If you overwrite {@link AbstractBlock#saveMetadata(DataOutputStream)} or need to provide
     * backward compatibility to older binary formats, you probably need to overwrite this. This
     * default implementation checks if the version number fits, if not it throws an {@link
     * MalformedModelException}. After that it restores the input shapes.
     *
     * @param loadVersion the version used for loading this metadata.
     * @param is the input stream we are loading from
     * @throws IOException loading failed
     * @throws MalformedModelException data can be loaded but has wrong format
     */
    protected void loadMetadata(byte loadVersion, DataInputStream is)
            throws IOException, MalformedModelException {
        if (loadVersion != version) {
            throw new MalformedModelException(
                    "Cannot load parameters for "
                            + this.getClass().getCanonicalName()
                            + ", expected version "
                            + version
                            + ", got "
                            + loadVersion
                            + ".");
        }
        readInputShapes(is);
    }

    protected void saveInputShapes(DataOutputStream os) throws IOException {
        os.writeInt(inputShapes.length);
        for (Shape shape : inputShapes) {
            os.write(shape.getEncoded());
        }
    }

    protected void readInputShapes(DataInputStream is) throws IOException {
        int len = is.readInt();
        Shape[] shapes = new Shape[len];
        for (int i = 0; i < len; ++i) {
            shapes[i] = Shape.decode(is);
        }
        if (inputShapes == null) {
            // load inputShapes from parameter file if Block has not been initialized
            inputShapes = shapes;
        }
    }

    /** {@inheritDoc} */
    @Override
    public String toString() {
        return Blocks.describe(this, null, 0);
    }

    /** {@inheritDoc} */
    @Override
    public Shape[] getInputShapes() {
        if (!isInitialized()) {
            throw new IllegalStateException(
                    "getInputShapes() can only be called after the initialization process");
        }
        return inputShapes;
    }

    /** {@inheritDoc} */
    @Override
    public DataType[] getOutputDataTypes() {
        return outputDataTypes;
    }
}
