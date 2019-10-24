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

package ai.djl.mxnet.nn;

import ai.djl.Device;
import ai.djl.mxnet.engine.CachedOp;
import ai.djl.mxnet.engine.MxNDManager;
import ai.djl.mxnet.engine.Symbol;
import ai.djl.mxnet.jna.JnaUtils;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Parameter;
import ai.djl.nn.ParameterBlock;
import ai.djl.nn.ParameterType;
import ai.djl.nn.SymbolBlock;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

// TODO: Need to add Memory management for all params
public class MxSymbolBlock extends ParameterBlock implements SymbolBlock {
    private static final byte VERSION = 2;

    private NDManager manager;
    private CachedOp op;
    private Symbol symbol;
    private List<Parameter> params; // includes input data
    private Map<String, Shape> paramShapes;
    private Shape[] outputShapes;

    public MxSymbolBlock(NDManager manager, Symbol symbol) {
        this.manager = manager;
        this.symbol = symbol;
        inputNames = new ArrayList<>();

        String[] allNames = symbol.getAllNames();
        params = new ArrayList<>(allNames.length);

        Set<String> auxNameSet = new HashSet<>(Arrays.asList(symbol.getAuxNames()));
        for (String name : allNames) {
            ParameterType type = inferType(name);
            boolean requireGrad = !auxNameSet.contains(name);

            params.add(new Parameter(name, this, type, requireGrad));
        }
    }

    public void setInputNames(List<String> inputNames) {
        this.inputNames = inputNames;
    }

    public List<Parameter> getAllParameters() {
        return params;
    }

    @Override
    public Shape[] initialize(
            NDManager manager, DataType dataType, Device[] devices, Shape[] inputShapes) {
        if (!initialized) {
            beforeInitialize(inputShapes);
            for (Parameter parameter : params) {
                if (!inputNames.contains(parameter.getName())) {
                    parameter.initialize(manager, dataType, inputShapes);
                }
            }
            initialized = true;
        }
        return getOutputShapes(manager, inputShapes);
    }

    /**
     * return layers' name.
     *
     * @return an List of String contains layers' nanme
     */
    public List<String> getLayerNames() {
        return symbol.getLayerNames();
    }

    /**
     * Returns the Symbolic graph from the model.
     *
     * @return {@link Symbol} object
     */
    public Symbol getSymbol() {
        return symbol;
    }

    @Override
    public PairList<String, Shape> describeInput() {
        PairList<String, Shape> inputData = new PairList<>();
        for (String name : inputNames) {
            inputData.add(name, new Shape());
        }
        return inputData;
    }

    @Override
    public NDList forward(
            ParameterStore parameterStore, NDList inputs, PairList<String, Object> params) {
        if (op == null) {
            op = JnaUtils.createCachedOp(this, (MxNDManager) manager);
        }
        return op.forward(parameterStore, inputs);
    }

    @Override
    public Shape[] getOutputShapes(NDManager manager, Shape[] inputShapes) {
        if (outputShapes == null) {
            String[] outputNames = symbol.getOutputNames();
            outputShapes = new Shape[outputNames.length];
            for (int i = 0; i < outputShapes.length; ++i) {
                outputShapes[i] = getParameterShape(outputNames[i], inputShapes);
            }
        }
        return outputShapes;
    }

    @Override
    public List<Parameter> getDirectParameters() {
        return params.stream()
                .filter(p -> !inputNames.contains(p.getName()))
                .collect(Collectors.toList());
    }

    @Override
    public void removeLastBlock() {
        List<String> layerNames = getLayerNames();
        String layerName = layerNames.get(layerNames.size() - 2);

        Symbol sliced = symbol.get(layerName);
        symbol.close();
        symbol = sliced;

        HashSet<String> set = new HashSet<>(Arrays.asList(symbol.getAllNames()));
        for (int i = params.size() - 1; i >= 0; --i) {
            Parameter parameter = params.get(i);
            if (!set.contains(parameter.getName())) {
                params.remove(i).close();
            }
        }
    }

    @Override
    public Shape getParameterShape(String name, Shape[] inputShapes) {
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

    @Override
    public void saveParameters(DataOutputStream os) throws IOException {
        os.writeByte(VERSION);
        int size = inputNames.size();
        os.writeInt(size);
        for (String name : inputNames) {
            os.writeUTF(name);
        }
        for (Parameter parameter : params) {
            if (!inputNames.contains(parameter.getName())) {
                parameter.save(os);
            }
        }
    }

    @Override
    public void loadParameters(NDManager manager, DataInputStream is) throws IOException {
        byte version = is.readByte();
        if (version != VERSION) {
            throw new IllegalArgumentException("Unsupported encoding version: " + version);
        }
        int size = is.readInt();
        for (int i = 0; i < size; ++i) {
            inputNames.add(is.readUTF());
        }

        for (Parameter parameter : params) {
            if (!inputNames.contains(parameter.getName())) {
                parameter.load(this.manager, is);
            }
        }
    }

    private static ParameterType inferType(String name) {
        if (name.endsWith("bias")) {
            return ParameterType.BIAS;
        } else if (name.endsWith("gamma")) {
            return ParameterType.GAMMA;
        } else if (name.endsWith("beta")) {
            return ParameterType.BETA;
        } else if (name.endsWith("moving_mean") || name.endsWith("running_mean")) {
            return ParameterType.RUNNING_MEAN;
        } else if (name.endsWith("moving_var") || name.endsWith("running_var")) {
            return ParameterType.RUNNING_VAR;
        }
        return ParameterType.OTHER;
    }
}
