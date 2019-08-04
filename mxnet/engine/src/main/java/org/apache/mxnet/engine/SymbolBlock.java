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

package org.apache.mxnet.engine;

import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;
import org.apache.mxnet.jna.JnaUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import software.amazon.ai.Block;
import software.amazon.ai.Parameter;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.ndarray.NDManager;
import software.amazon.ai.ndarray.types.DataDesc;
import software.amazon.ai.ndarray.types.DataType;
import software.amazon.ai.ndarray.types.Shape;
import software.amazon.ai.util.PairList;

// TODO: Need to add Memory management for all params
public class SymbolBlock implements Block {

    private static final Logger logger = LoggerFactory.getLogger(MxModel.class);

    private CachedOp op;
    private Symbol symbol;
    private MxNDManager manager;
    private List<Parameter> params;

    public SymbolBlock(Symbol symbol, List<Parameter> params, NDManager manager) {
        this.symbol = symbol;
        this.params = params;
        this.manager = (MxNDManager) manager;
    }

    @Override
    public DataDesc[] describeInput() {
        String[] allNames = symbol.getAllNames();
        Map<String, Integer> map = new ConcurrentHashMap<>(allNames.length * 3 / 2);
        List<String> paramNames =
                params.stream().map(Parameter::getName).collect(Collectors.toList());
        int index = 0;
        for (String name : allNames) {
            map.put(name, index++);
        }
        for (String name : paramNames) {
            map.remove(name);
        }
        DataDesc[] inputData = new DataDesc[map.size()];

        index = 0;
        for (String name : map.keySet()) {
            inputData[index++] = new DataDesc(new Shape(), name);
        }
        return inputData;
    }

    @Override
    public SymbolBlock cast(DataType dataType) {
        if (params.get(0).getArray().getDataType() == dataType) {
            logger.debug("You are casting the model to its original type!");
            return this;
        }

        // TODO: This implementation is unsafe, new SymbolBlock shares the same
        // symbol and optimizerStates with original one. Close either one
        // will cause anther SymbolBlock instance invalidated.
        List<Parameter> newParams =
                params.stream()
                        .map(
                                ele -> {
                                    NDArray casted = ele.getArray().asType(dataType, true);
                                    return new Parameter(ele.getName(), casted);
                                })
                        .collect(Collectors.toList());
        return new SymbolBlock(symbol, newParams, manager);
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
    public NDList forward(NDList inputs, PairList<String, Object> params) {
        // TODO: if user would like to reinitialize param and create cachedOp
        // TODO: the existing one won't update
        if (op == null) {
            op = JnaUtils.createCachedOp(this, manager);
        }
        return op.forward(inputs);
    }

    @Override
    public void backward() {}

    @Override
    public boolean isInitialized() {
        return true;
    }

    @Override
    public Shape getOutputShape(Shape... inputs) {
        return null;
    }

    @Override
    public List<Parameter> getDirectParameters() {
        return params;
    }

    @Override
    public void beforeInitialize(NDList inputs) {}

    @Override
    public Shape getParameterShape(String name, NDList inputs) {
        for (Parameter param : params) {
            if (param.getName().equals(name)) {
                return param.getArray().getShape();
            }
        }
        throw new IllegalArgumentException("Name " + name + " not found");
    }

    @Override
    public byte[] getEncoded() {
        return new byte[0];
    }
}
