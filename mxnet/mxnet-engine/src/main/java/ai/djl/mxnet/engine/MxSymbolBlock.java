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

package ai.djl.mxnet.engine;

import ai.djl.MalformedModelException;
import ai.djl.mxnet.jna.JnaUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractSymbolBlock;
import ai.djl.nn.Parameter;
import ai.djl.nn.SymbolBlock;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * {@code MxSymbolBlock} is the MXNet implementation of {@link SymbolBlock}.
 *
 * <p>You can create a {@code MxSymbolBlock} using {@link ai.djl.Model#load(java.nio.file.Path,
 * String)}.
 */
public class MxSymbolBlock extends AbstractSymbolBlock {

    private static final Logger logger = LoggerFactory.getLogger(MxSymbolBlock.class);

    private static final byte VERSION = 3;

    private NDManager manager;
    private CachedOp op;
    private Symbol symbol;
    private List<Parameter> mxNetParams; // includes input data
    private Map<String, Shape> paramShapes;
    private Shape[] outputShapes;
    private PairList<String, Shape> inputDescriptions;
    private PairList<String, Shape> outputDescriptions;
    private boolean first;

    /**
     * Constructs a {@code MxSymbolBlock} for a {@link Symbol}.
     *
     * <p>You can create a {@code MxSymbolBlock} using {@link ai.djl.Model#load(java.nio.file.Path,
     * String)}.
     *
     * @param manager the manager to use for the block
     * @param symbol the symbol containing the block's symbolic graph
     */
    public MxSymbolBlock(NDManager manager, Symbol symbol) {
        this(manager);
        this.symbol = symbol;
        initBlock();
    }

    /**
     * Constructs an empty {@code MxSymbolBlock}.
     *
     * @param manager the manager to use for the block
     */
    public MxSymbolBlock(NDManager manager) {
        super(VERSION);
        this.manager = manager;
    }

    /**
     * Sets the names of the input data.
     *
     * @param inputNames the names of the input data
     */
    public void setInputNames(List<String> inputNames) {
        this.inputNames = inputNames;
        // now that we know which of the parameters are just input placeholders and which
        // are trainable, add them properly so they are correctly handled
        Set<String> nameLookup = new HashSet<>(inputNames);
        for (Parameter mxNetParameter : mxNetParams) {
            if (!nameLookup.contains(mxNetParameter.getName())) {
                addParameter(mxNetParameter);
            }
        }
    }

    /**
     * Returns the list of inputs and parameter NDArrays.
     *
     * @return the list of inputs and parameter NDArrays
     */
    public List<Parameter> getAllParameters() {
        return mxNetParams;
    }

    /**
     * Returns the layers' name.
     *
     * @return a List of String containing the layers' name
     */
    public List<String> getLayerNames() {
        return symbol.getLayerNames();
    }

    /**
     * Returns the Symbolic graph from the model.
     *
     * @return a {@link Symbol} object
     */
    public Symbol getSymbol() {
        return symbol;
    }

    /**
     * Applies Optimization algorithm for the model.
     *
     * @param optimization the name of the optimization
     */
    public void optimizeFor(String optimization) {
        Symbol newSymbol = symbol.optimizeFor(optimization, manager.getDevice());
        symbol.close();
        symbol = newSymbol;
    }

    /** {@inheritDoc} */
    @Override
    public PairList<String, Shape> describeInput() {
        if (inputDescriptions == null) {
            inputDescriptions = new PairList<>();
            for (String name : inputNames) {
                // Add empty shapes as input shapes are not saved
                // in MXNet models
                logger.warn(
                        "Input shapes are unknown, please run predict or forward once"
                                + "and call describeInput again.");
                inputDescriptions.add(name, new Shape());
            }
        }
        return inputDescriptions;
    }

    /** {@inheritDoc} */
    @Override
    public PairList<String, Shape> describeOutput() {
        if (outputDescriptions == null) {
            logger.warn(
                    "Output shapes are unknown, please run predict or forward once"
                            + "and call describeOutput again.");
        }
        return outputDescriptions;
    }

    /** {@inheritDoc} */
    @Override
    protected NDList forwardInternal(
            ParameterStore parameterStore,
            NDList inputs,
            boolean training,
            PairList<String, Object> params) {
        if (first) {
            synchronized (MxSymbolBlock.class) {
                if (first) {
                    // create CachedOp is not thread-safe
                    // add synchronized block to avoid creating multiple CachedOps
                    op = JnaUtils.createCachedOp(this, (MxNDManager) manager, training);
                    inputDescriptions = new PairList<>();
                    outputDescriptions = new PairList<>();
                    for (NDArray array : inputs) {
                        inputDescriptions.add(array.getName(), array.getShape());
                    }
                    NDList outputs = op.forward(parameterStore, inputs, training);
                    for (NDArray array : outputs) {
                        outputDescriptions.add(array.getName(), array.getShape());
                    }
                    first = false;
                    return outputs;
                }
            }
        }
        return op.forward(parameterStore, inputs, training);
    }

    /** {@inheritDoc} */
    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        if (outputShapes == null) {
            String[] outputNames = symbol.getOutputNames();
            outputShapes = new Shape[outputNames.length];
            for (int i = 0; i < outputShapes.length; ++i) {
                outputShapes[i] = getParameterShape(outputNames[i], inputShapes);
            }
        }
        return outputShapes;
    }

    /** {@inheritDoc} */
    @Override
    public void removeLastBlock() {
        List<String> layerNames = getLayerNames();
        String layerName = layerNames.get(layerNames.size() - 2);

        Symbol sliced = symbol.get(layerName);
        symbol.close();
        symbol = sliced;

        HashSet<String> set = new HashSet<>(Arrays.asList(symbol.getAllNames()));
        for (int i = mxNetParams.size() - 1; i >= 0; --i) {
            Parameter parameter = mxNetParams.get(i);
            if (!set.contains(parameter.getName())) {
                mxNetParams.remove(i).close();
                parameters.remove(parameter.getName(), parameter);
            }
        }
    }

    private Shape getParameterShape(String name, Shape[] inputShapes) {
        if (paramShapes == null) {
            PairList<String, Shape> pairs = new PairList<>();
            for (int i = 0; i < inputNames.size(); i++) {
                pairs.add(inputNames.get(i), inputShapes[i]);
            }
            paramShapes = symbol.inferShape(pairs);
        }
        if (paramShapes.containsKey(name)) {
            return paramShapes.get(name);
        } else {
            throw new IllegalArgumentException("Name " + name + " not found");
        }
    }

    /** {@inheritDoc} */
    @Override
    public void saveParameters(DataOutputStream os) throws IOException {
        os.writeByte(VERSION);
        String json = symbol.toJsonString();
        // symbol size may go beyond os.writeUTF() size (65535)
        byte[] bytes = json.getBytes(StandardCharsets.UTF_8);
        os.writeInt(bytes.length);
        os.write(bytes);
        int size = inputNames.size();
        os.writeInt(size);
        for (String name : inputNames) {
            os.writeUTF(name);
        }
        for (Parameter parameter : mxNetParams) {
            parameter.save(os);
        }
    }

    /** {@inheritDoc} */
    @Override
    public void loadParameters(NDManager manager, DataInputStream is)
            throws IOException, MalformedModelException {
        byte version = is.readByte();
        if (version > VERSION) {
            throw new MalformedModelException("Unsupported encoding version: " + version);
        }
        if (version < VERSION && symbol == null) {
            throw new IllegalStateException(
                    "Symbol is required for version 2, please use Model to load");
        }
        if (version == VERSION) {
            int len = is.readInt();
            byte[] bytes = new byte[len];
            if (is.read(bytes) == -1) {
                throw new MalformedModelException("InputStream ends at symbol loading!");
            }
            // init block only if it is not set
            symbol =
                    Symbol.loadJson(
                            (MxNDManager) manager, new String(bytes, StandardCharsets.UTF_8));
            initBlock();
        }
        int size = is.readInt();
        for (int i = 0; i < size; ++i) {
            inputNames.add(is.readUTF());
        }

        for (Parameter parameter : mxNetParams) {
            parameter.load(this.manager, is);
        }
        setInputNames(inputNames);
    }

    private void initBlock() {
        inputNames = new ArrayList<>();

        String[] allNames = symbol.getAllNames();
        mxNetParams = new ArrayList<>(allNames.length);

        Set<String> auxNameSet = new HashSet<>(Arrays.asList(symbol.getAuxNames()));
        for (String name : allNames) {
            Parameter.Type type = inferType(name);
            boolean requireGrad = !auxNameSet.contains(name);
            mxNetParams.add(
                    Parameter.builder()
                            .setName(name)
                            .setType(type)
                            .optRequiresGrad(requireGrad)
                            .build());
        }
        first = true;
    }

    private static Parameter.Type inferType(String name) {
        if (name.endsWith("bias")) {
            return Parameter.Type.BIAS;
        } else if (name.endsWith("gamma")) {
            return Parameter.Type.GAMMA;
        } else if (name.endsWith("beta")) {
            return Parameter.Type.BETA;
        } else if (name.endsWith("moving_mean") || name.endsWith("running_mean")) {
            return Parameter.Type.RUNNING_MEAN;
        } else if (name.endsWith("moving_var") || name.endsWith("running_var")) {
            return Parameter.Type.RUNNING_VAR;
        } else if (name.endsWith("weight")) {
            return Parameter.Type.WEIGHT;
        }
        return Parameter.Type.OTHER;
    }
}
