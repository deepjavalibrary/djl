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
import ai.djl.training.initializer.Initializer;
import ai.djl.util.Pair;
import ai.djl.util.PairList;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.function.Predicate;

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
 *       AbstractBlock#loadMetadata(byte, DataInputStream)}. If you do not need to save or load any
 *       state other than parameters in your block, you can skip this.
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
public abstract class AbstractBlock implements Block {

    /** The shape of the input for this block, set by the initialization process. */
    protected Shape[] inputShapes;

    /** List of names for the input, named inputs should be manually set in sub class. */
    protected List<String> inputNames = Collections.emptyList();

    /**
     * The model version of this block, used for checking if parameters are still valid during
     * parameter loading.
     */
    protected byte version;

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
    public AbstractBlock() {
        this((byte) 1);
    }

    /**
     * Builds an empty block with the given version for parameter serialization.
     *
     * @param version the version to use for parameter serialization.
     */
    public AbstractBlock(byte version) {
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

    /**
     * Use this to add a child block to this block.
     *
     * @param name Name of the block, must be unique or otherwise existing children with this name
     *     are removed, must not be null.
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
     * Adds a parameter to this block. If parameters are added with this method, intialization of
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
        Parameter parameter = parameters.get(paramName);
        if (parameter == null) {
            throw new IllegalArgumentException("Could not find parameter " + paramName);
        }
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
        // if parameters are initialized, skip it
        if (!isInitialized()) {
            // setShape for all params
            prepare(inputShapes);
        }
        for (Parameter parameter : parameters.values()) {
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
        if (!children.isEmpty()) {
            throw new IllegalStateException(
                    getClass().getSimpleName()
                            + " has child blocks but initializeChildBlocks is not overwritten.");
        }
    }

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
    public ParameterList getDirectParameters() {
        return new ParameterList(parameters);
    }

    /**
     * Sets the shape of {@link Parameter}s.
     *
     * @param inputShapes the shapes of inputs
     */
    protected void prepare(Shape[] inputShapes) {}

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
        for (Parameter parameter : parameters.values()) {
            parameter.save(os);
        }
        for (Block child : children.values()) {
            child.saveParameters(os);
        }
    }

    /** {@inheritDoc} */
    @Override
    public void loadParameters(NDManager manager, DataInputStream is)
            throws IOException, MalformedModelException {
        byte loadVersion = is.readByte();
        loadMetadata(loadVersion, is);
        for (Parameter parameter : parameters.values()) {
            parameter.load(manager, is);
        }
        for (Block child : children.values()) {
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
     * backward compatibility to older binary formats, you prabably need to overwrite this. This
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
        // FIXME: This is a quick hack for display in jupyter notebook.
        StringBuilder sb = new StringBuilder(200);
        String className = getClass().getSimpleName();
        if (className.endsWith("Block")) {
            className = className.substring(0, className.length() - 5);
        }
        sb.append(className).append('(');
        if (isInitialized()) {
            PairList<String, Shape> inputShapeDescription = describeInput();
            appendShape(sb, inputShapeDescription.values().toArray(new Shape[0]));
            sb.append(" -> ");
            Shape[] outputShapes =
                    getOutputShapes(inputShapeDescription.values().toArray(new Shape[0]));
            appendShape(sb, outputShapes);
        } else {
            sb.append("Uninitialized");
        }
        sb.append(')');
        return sb.toString();
    }

    private void appendShape(StringBuilder sb, Shape[] shapes) {
        boolean first = true;
        for (Shape shape : shapes) {
            if (first) {
                first = false;
            } else {
                sb.append(", ");
            }
            long[] sh = shape.getShape();
            int length = sh.length;
            if (length == 0) {
                sb.append("()");
            } else {
                int index = 0;
                if (sh[0] == -1) {
                    --length;
                    index = 1;
                }

                if (length == 0) {
                    sb.append("()");
                } else if (length == 1) {
                    sb.append(sh[index]);
                } else {
                    sb.append('(');
                    for (int i = index; i < sh.length; ++i) {
                        if (i > index) {
                            sb.append(", ");
                        }
                        sb.append(sh[i]);
                    }
                    sb.append(')');
                }
            }
        }
    }
}
