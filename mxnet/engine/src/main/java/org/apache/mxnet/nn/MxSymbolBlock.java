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

package org.apache.mxnet.nn;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.apache.mxnet.engine.CachedOp;
import org.apache.mxnet.engine.MxModel;
import org.apache.mxnet.engine.MxNDManager;
import org.apache.mxnet.engine.Symbol;
import org.apache.mxnet.jna.JnaUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.ndarray.NDManager;
import software.amazon.ai.ndarray.types.DataDesc;
import software.amazon.ai.ndarray.types.DataType;
import software.amazon.ai.ndarray.types.Shape;
import software.amazon.ai.nn.Parameter;
import software.amazon.ai.nn.ParameterBlock;
import software.amazon.ai.nn.ParameterType;
import software.amazon.ai.nn.SymbolBlock;
import software.amazon.ai.util.PairList;

// TODO: Need to add Memory management for all params
public class MxSymbolBlock extends ParameterBlock implements SymbolBlock {

    private static final Logger logger = LoggerFactory.getLogger(MxModel.class);

    private static final byte VERSION = 1;

    private NDManager manager;
    private CachedOp op;
    private Symbol symbol;
    private List<Parameter> params;
    private List<String> inputNames;
    private Map<String, Shape> paramShapes;
    private Shape[] outputShapes;

    public MxSymbolBlock(NDManager manager, Symbol symbol, List<String> names) {
        this.manager = manager;
        this.symbol = symbol;
        this.inputNames = names;
        params = new ArrayList<>();

        Set<String> inputNameSet = new HashSet<>(names);
        Set<String> auxNameSet = new HashSet<>(Arrays.asList(symbol.getAuxNames()));

        String[] allNames = symbol.getAllNames();
        for (String name : allNames) {
            if (inputNameSet.contains(name)) {
                continue;
            }
            ParameterType type = inferType(name);
            boolean requireGrad = !auxNameSet.contains(name);

            params.add(new Parameter(name, this, type, requireGrad));
        }
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

    private List<String> getParamNames() {
        List<String> nameList = Arrays.asList(symbol.getAllNames());
        inputNames.forEach(nameList::remove);
        return nameList;
    }

    @Override
    public DataDesc[] describeInput() {
        DataDesc[] inputData = new DataDesc[inputNames.size()];
        int index = 0;
        for (String name : inputNames) {
            inputData[index++] = new DataDesc(new Shape(), name);
        }
        return inputData;
    }

    @Override
    public void cast(DataType dataType) {
        if (params.get(0).getArray().getDataType() == dataType) {
            logger.debug("You are casting the model to its original type!");
            return;
        }

        // TODO: Not all parameters can be casted.
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    @Override
    public NDList forward(NDList inputs, PairList<String, Object> params) {
        if (op == null) {
            op = JnaUtils.createCachedOp(this, (MxNDManager) manager);
        }
        return op.forward(inputs);
    }

    @Override
    public void backward() {}

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
        return params;
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
        for (Parameter parameter : params) {
            parameter.save(os);
        }
    }

    @Override
    public void loadParameters(NDManager manager, DataInputStream is) throws IOException {
        byte version = is.readByte();
        if (version != VERSION) {
            throw new IllegalArgumentException("Unsupported encoding version: " + version);
        }
        for (Parameter parameter : params) {
            parameter.load(this.manager, is);
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
