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

import ai.djl.Device;
import ai.djl.mxnet.jna.JnaUtils;
import ai.djl.ndarray.types.Shape;
import ai.djl.util.NativeResource;
import ai.djl.util.PairList;
import ai.djl.util.Utils;
import com.sun.jna.Pointer;
import java.util.Arrays;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

/**
 * {@code Symbol} is an internal helper for symbolic model graphs used by the {@link
 * ai.djl.nn.SymbolBlock}.
 *
 * @see ai.djl.nn.SymbolBlock
 * @see <a href="https://mxnet.incubator.apache.org/api/python/docs/api/symbol/index.html">MXNet
 *     Symbol</a>
 */
public class Symbol extends NativeResource<Pointer> {

    //    private String[] argParams;
    //    private String[] auxParams;
    private String[] outputs;
    //    private List<Integer> outputLayouts;
    private MxNDManager manager;

    /**
     * Constructs a {@code Symbol}.
     *
     * @param manager the manager to attach the symbol to
     * @param pointer the symbol's native data location
     */
    Symbol(MxNDManager manager, Pointer pointer) {
        super(pointer);
        this.manager = manager;
        manager.attachInternal(getUid(), this);
        //        argParams = JnaUtils.listSymbolArguments(getHandle());
        //        auxParams = JnaUtils.listSymbolAuxiliaryStates(getHandle());
    }

    /**
     * Loads a symbol from a path.
     *
     * @param manager the manager to load the symbol to
     * @param path the path to the symbol file
     * @return the new symbol
     */
    public static Symbol load(MxNDManager manager, String path) {
        Pointer pointer = JnaUtils.createSymbolFromFile(path);
        return new Symbol(manager, pointer);
    }

    /**
     * Loads a symbol from a json string.
     *
     * @param manager the manager to load the symbol to
     * @param json the json string of the symbol.
     * @return the new symbol
     */
    public static Symbol loadJson(MxNDManager manager, String json) {
        Pointer pointer = JnaUtils.createSymbolFromString(json);
        return new Symbol(manager, pointer);
    }

    /**
     * Returns the symbol argument names.
     *
     * @return the symbol argument names
     */
    public String[] getArgNames() {
        return JnaUtils.listSymbolArguments(getHandle());
    }

    /**
     * Returns the MXNet auxiliary states for the symbol.
     *
     * @return the MXNet auxiliary states for the symbol
     */
    public String[] getAuxNames() {
        return JnaUtils.listSymbolAuxiliaryStates(getHandle());
    }

    /**
     * Returns the symbol names.
     *
     * @return the symbol names
     */
    public String[] getAllNames() {
        return JnaUtils.listSymbolNames(getHandle());
    }

    /**
     * Returns the symbol outputs.
     *
     * @return the symbol outputs
     */
    public String[] getOutputNames() {
        if (outputs == null) {
            outputs = JnaUtils.listSymbolOutputs(getHandle());
        }
        return outputs;
    }

    private String[] getInternalOutputNames() {
        return JnaUtils.listSymbolOutputs(getInternals().getHandle());
    }

    /*
    public List<Integer> getOutputLayouts() {
        if (outputLayouts == null) {
            outputLayouts = new ArrayList<>();
            for (String argName : getArgParams()) {
                try (Symbol symbol = get(argName)) {
                    Layout layout = Layout.fromValue(symbol.getAttribute("__layout__"));
                    outputLayouts.add(DataDesc.getBatchAxis(layout));
                }
            }
        }
        return outputLayouts;
    }

    public String getAttribute(String key) {
        return JnaUtils.getSymbolAttr(getHandle(), key);
    }

    public PairList<String, String> getAttributes() {
        return JnaUtils.listSymbolAttr(getHandle());
    }

     */

    /**
     * Copies the symbol.
     *
     * @return a new copy of the symbol
     */
    public Symbol copy() {
        throw new UnsupportedOperationException("Not implemented yet");
    }

    /**
     * Returns the output symbol by index.
     *
     * @param index the index of the output
     * @return the symbol output as a new symbol
     */
    public Symbol get(int index) {
        Pointer pointer = JnaUtils.getSymbolOutput(getInternals().getHandle(), index);
        return new Symbol(manager, pointer);
    }

    /**
     * Returns the output symbol with the given name.
     *
     * @param name the name of the symbol to return
     * @return the output symbol
     * @throws IllegalArgumentException Thrown if no output matches the name
     */
    public Symbol get(String name) {
        String[] out = getInternalOutputNames();
        int index = Utils.indexOf(out, name);
        if (index < 0) {
            throw new IllegalArgumentException("Cannot find output that matches name: " + name);
        }
        return get(index);
    }

    /**
     * Returns the symbol internals.
     *
     * @return the symbol internals symbol
     */
    public Symbol getInternals() {
        Pointer pointer = JnaUtils.getSymbolInternals(getHandle());
        return new Symbol(manager, pointer);
    }

    /**
     * Returns the list of names for all internal outputs.
     *
     * @return a list of names
     */
    public List<String> getLayerNames() {
        String[] outputNames = getInternalOutputNames();
        String[] allNames = getAllNames();
        Set<String> allNamesSet = new LinkedHashSet<>(Arrays.asList(allNames));
        // Kill all params field and keep the output layer
        return Arrays.stream(outputNames)
                .filter(n -> !allNamesSet.contains(n))
                .collect(Collectors.toList());
    }

    /**
     * Infers the shapes for all parameters inside a symbol from the given input shapes.
     *
     * @param pairs the given input name and shape
     * @return a map of arguments with names and shapes
     */
    public Map<String, Shape> inferShape(PairList<String, Shape> pairs) {
        List<List<Shape>> shapes = JnaUtils.inferShape(this, pairs);
        if (shapes == null) {
            throw new IllegalArgumentException("Cannot infer shape based on the data provided!");
        }
        List<Shape> argShapes = shapes.get(0);
        List<Shape> outputShapes = shapes.get(1);
        List<Shape> auxShapes = shapes.get(2);
        // TODO: add output to the map
        String[] argNames = getArgNames();
        String[] auxNames = getAuxNames();
        String[] outputNames = getOutputNames();
        Map<String, Shape> shapesMap = new ConcurrentHashMap<>();
        for (int i = 0; i < argNames.length; i++) {
            shapesMap.put(argNames[i], argShapes.get(i));
        }
        for (int i = 0; i < auxNames.length; i++) {
            shapesMap.put(auxNames[i], auxShapes.get(i));
        }
        for (int i = 0; i < outputNames.length; i++) {
            shapesMap.put(outputNames[i], outputShapes.get(i));
        }
        return shapesMap;
    }

    /**
     * [Experimental] Add customized optimization on the Symbol.
     *
     * <p>This method can be used with EIA or TensorRT for model acceleration
     *
     * @param backend backend name
     * @param device the device assigned
     * @return optimized Symbol
     */
    public Symbol optimizeFor(String backend, Device device) {
        return new Symbol(manager, JnaUtils.optimizeFor(this, backend, device));
    }

    /*

    public String debugStr() {
        return JnaUtils.getSymbolDebugString(getHandle());
    }

    public void setAttr(Map<String, String> attrs) {
        for (Map.Entry<String, String> entry : attrs.entrySet()) {
            JnaUtils.setSymbolAttr(getHandle(), entry.getKey(), entry.getValue());
        }
    }

    public PairList<String, String> listAttr() {
        return JnaUtils.listSymbolAttr(getHandle());
    }

    public PairList<String, String> attrMap() {
        return JnaUtils.listSymbolAttr(getHandle());
    }

    public void save(String path) {
        JnaUtils.saveSymbol(getHandle(), path);
    }

    public Symbol compose(String name, String[] keys) {
        return new Symbol(manager, JnaUtils.compose(getHandle(), name, keys));
    }

    public void compose(String name, Map<String, String> symbols) {
        JnaUtils.compose(getHandle(), name, symbols.values().toArray(JnaUtils.EMPTY_ARRAY));
    }

    public String toJson() {
        return JnaUtils.symbolToJson(getHandle());
    }

     */

    /**
     * Converts Symbol to json string for saving purpose.
     *
     * @return the json string
     */
    public String toJsonString() {
        return JnaUtils.getSymbolString(getHandle());
    }

    /** {@inheritDoc} */
    @Override
    public String toString() {
        return Arrays.toString(getOutputNames());
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        Pointer pointer = handle.getAndSet(null);
        if (pointer != null) {
            manager.detachInternal(getUid());
            JnaUtils.freeSymbol(pointer);
            manager = null;
        }
    }
}
