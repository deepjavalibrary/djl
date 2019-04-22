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
package org.apache.mxnet.model;

import com.amazon.ai.util.PairList;
import com.amazon.ai.util.Utils;
import com.sun.jna.Pointer;
import com.sun.jna.ptr.PointerByReference;
import java.nio.IntBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import org.apache.mxnet.Context;
import org.apache.mxnet.jna.JnaUtils;
import org.apache.mxnet.jna.MxnetLibrary;
import org.apache.mxnet.types.DataType;
import org.apache.mxnet.types.GradReq;
import org.apache.mxnet.types.StorageType;

public class Symbol extends NativeResource {

    private String[] argParams;
    private String[] auxParams;
    private String[] outputs;
    private List<Integer> outputLayouts;

    Symbol(ResourceAllocator alloc, Pointer pointer) {
        super(alloc, pointer);
        argParams = JnaUtils.listSymbolArguments(getHandle());
        auxParams = JnaUtils.listSymbolAuxiliaryStates(getHandle());
    }

    public static Symbol load(ResourceAllocator alloc, String path) {
        Pointer pointer = JnaUtils.createSymbolFromFile(path);
        return new Symbol(alloc, pointer);
    }

    public String[] getArgParams() {
        return argParams;
    }

    public String[] getAuxParams() {
        return auxParams;
    }

    public String[] getOutputs() {
        if (outputs == null) {
            outputs = JnaUtils.listSymbolOutputs(getHandle());
        }
        return outputs;
    }

    public List<Integer> getOutputLayouts() {
        if (outputLayouts == null) {
            outputLayouts = new ArrayList<>();
            for (String argName : getArgParams()) {
                try (Symbol symbol = get(argName)) {
                    String layout = symbol.getAttribute("__layout__");
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

    public void plus(Symbol other) {}

    public void plusScalar(Symbol other) {}

    public void minus(Symbol other) {}

    public void minusScalar(Symbol other) {}

    public Symbol copy() {
        return this;
    }

    public Symbol get(int index) {
        Pointer pointer = JnaUtils.getSymbolOutput(getHandle(), index);
        return new Symbol(alloc, pointer);
    }

    public Symbol get(String name) {
        String[] out = getOutputs();
        int index = Utils.indexOf(out, name);
        if (index < 0) {
            throw new IllegalArgumentException("Cannot find output that matches name: " + name);
        }
        return get(index);
    }

    public Symbol getInternals() {
        Pointer pointer = JnaUtils.getSymbolInternals(getHandle());
        return new Symbol(alloc, pointer);
    }

    public void inferType(DataType... args) {
        String[] types = new String[args.length];
        int i = 0;
        for (DataType type : args) {
            types[i++] = type.getType();
        }
        JnaUtils.inferType(getHandle(), types);
    }

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
        return new Symbol(alloc, JnaUtils.compose(getHandle(), name, keys));
    }

    public void compose(String name, Map<String, String> symbols) {
        JnaUtils.compose(getHandle(), name, symbols.values().toArray(new String[0]));
    }

    public MxExecutor[] simpleBind(
            MxModel model,
            List<Context> contexts,
            List<DataDesc> dataDescriptors,
            String[] labelNames,
            String[] stateNames,
            GradReq gradReq,
            Map<String, Context> g2cMap,
            Map<String, StorageType> stypeMap) {
        MxExecutor[] executors = new MxExecutor[contexts.size()];

        // each argParams have a gradReq value
        String[] argParamGradReqs = new String[argParams.length];
        Arrays.fill(argParamGradReqs, gradReq.getType());

        // g2c
        String[] g2cKeys = null;
        int[] g2cDeviceTypes = null;
        int[] g2cDeviceIds = null;
        if (g2cMap != null && !g2cMap.isEmpty()) {
            g2cKeys = new String[g2cMap.size()];
            g2cDeviceTypes = new int[g2cKeys.length];
            g2cDeviceIds = new int[g2cKeys.length];

            int k = 0;
            for (Map.Entry<String, Context> entry : g2cMap.entrySet()) {
                g2cKeys[k] = entry.getKey();
                Context ctx = entry.getValue();
                g2cDeviceTypes[k] = ctx.getDeviceType().getType();
                g2cDeviceIds[k] = ctx.getDeviceId();
                ++k;
            }
        }

        // Prepare input data related parameters
        int size = 0;
        for (DataDesc desc : dataDescriptors) {
            size += desc.getShape().getShape().length;
        }
        String[] inputArgNames = new String[dataDescriptors.size()];
        String[] inputDataTypeNames = new String[inputArgNames.length];
        int[] inputDataTypes = new int[inputArgNames.length];

        IntBuffer inputShapeData = IntBuffer.allocate(size);
        IntBuffer inputShapeIdx = IntBuffer.allocate(inputArgNames.length + 1);
        inputShapeIdx.put(0);
        int k = 0;
        int offset = 0;
        for (DataDesc desc : dataDescriptors) {
            inputArgNames[k] = desc.getName();
            inputDataTypeNames[k] = desc.getName();
            inputDataTypes[k] = desc.getDataType().ordinal();
            int[] shape = desc.getShape().getShape();
            inputShapeData.put(shape);
            offset += shape.length;
            inputShapeIdx.put(offset);
            ++k;
        }
        inputShapeData.rewind();
        inputShapeIdx.rewind();

        String[] inputStorageTypeNames = null;
        int[] inputStorageTypes = null;
        if (stypeMap != null && !stypeMap.isEmpty()) {
            inputStorageTypeNames = new String[stypeMap.size()];
            inputStorageTypes = new int[inputStorageTypeNames.length];

            k = 0;
            for (Map.Entry<String, StorageType> entry : stypeMap.entrySet()) {
                inputStorageTypeNames[k] = entry.getKey();
                inputStorageTypes[k] = entry.getValue().getValue();
                ++k;
            }
        }

        // filter argParams from inputNames, labelNames, and stateNames
        List<String> sharedArgNames = new ArrayList<>();
        for (String arg : argParams) {
            if (!Utils.contains(inputArgNames, arg)
                    && !Utils.contains(labelNames, arg)
                    && !Utils.contains(stateNames, arg)) {
                sharedArgNames.add(arg);
            }
        }
        String[] sharedArgParams = sharedArgNames.toArray(new String[0]); // NOPMD

        IntBuffer sharedBufferLen = IntBuffer.allocate(1);
        sharedBufferLen.put(0, 0);
        String[] sharedBufferNames = new String[0];
        PointerByReference sharedBufferHandles = new PointerByReference();

        for (int i = 0; i < contexts.size(); ++i) {
            Context context = contexts.get(i);

            int deviceId = context.getDeviceId();
            int deviceType = context.getDeviceType().getType();

            PointerByReference updatedSharedBufferNames = new PointerByReference();
            PointerByReference updatedSharedBufferHandles = new PointerByReference();

            IntBuffer numInArgs = IntBuffer.allocate(1);
            PointerByReference inArgs = new PointerByReference();
            PointerByReference argGrads = new PointerByReference();
            IntBuffer numAuxStates = IntBuffer.allocate(1);
            PointerByReference auxStates = new PointerByReference();
            PointerByReference ref = new PointerByReference();

            JnaUtils.checkCall(
                    MxnetLibrary.INSTANCE.MXExecutorSimpleBind(
                            getHandle(),
                            deviceType,
                            deviceId,
                            g2cKeys == null ? 0 : g2cKeys.length,
                            g2cKeys,
                            g2cDeviceTypes,
                            g2cDeviceIds,
                            argParams.length,
                            argParams,
                            argParamGradReqs,
                            inputArgNames.length,
                            inputArgNames,
                            inputShapeData.array(),
                            inputShapeIdx.array(),
                            inputDataTypeNames.length,
                            inputDataTypeNames,
                            inputDataTypes,
                            inputStorageTypeNames == null ? 0 : inputStorageTypeNames.length,
                            inputStorageTypeNames,
                            inputStorageTypes,
                            sharedArgParams.length,
                            sharedArgParams,
                            sharedBufferLen,
                            sharedBufferNames,
                            sharedBufferHandles,
                            updatedSharedBufferNames,
                            updatedSharedBufferHandles,
                            numInArgs,
                            inArgs,
                            argGrads,
                            numAuxStates,
                            auxStates,
                            null,
                            ref));

            // update shared buffer
            int updatedSize = sharedBufferLen.get(0);
            if (updatedSize > 0) {
                Pointer[] updatedPointer =
                        updatedSharedBufferHandles.getValue().getPointerArray(0, updatedSize);
                String[] updatedNames =
                        updatedSharedBufferNames.getValue().getStringArray(0, updatedSize);
            }

            Map<String, NdArray> argParamMap = model.getArgParams().toMap();

            // get output for current executor's in_args, arg_grads, and aux_states
            int inArgSize = numInArgs.get(0);
            Pointer[] inArgsPointers = inArgs.getValue().getPointerArray(0, inArgSize);
            Pointer[] gradPointers = argGrads.getValue().getPointerArray(0, inArgSize);
            NdArray[] argArray = new NdArray[inArgSize];
            NdArray[] gradArray = new NdArray[inArgSize];
            NdArray[] dataArray = new NdArray[inputArgNames.length];
            int dataIdx = 0;
            for (int j = 0; j < inArgSize; ++j) {
                argArray[j] = new NdArray(alloc, inArgsPointers[j]);

                String paramName = argParams[j];

                NdArray param = argParamMap.get(paramName);
                if (param == null) {
                    if (Utils.contains(inputArgNames, paramName)) {
                        dataArray[dataIdx++] = argArray[j];
                    }
                } else {
                    param.copyTo(argArray[j]);
                }

                if (gradPointers[j] != null) {
                    gradArray[j] = new NdArray(alloc, gradPointers[j]);
                }
            }

            int auxStatesSize = numAuxStates.get();
            NdArray[] auxArray = new NdArray[auxStatesSize];
            if (auxStatesSize > 0) {
                Map<String, NdArray> auxParamMap = model.getAuxParams().toMap();
                Pointer[] pointers = auxStates.getValue().getPointerArray(0, auxStatesSize);
                for (int j = 0; j < auxStatesSize; ++j) {
                    auxArray[j] = new NdArray(alloc, pointers[j]);

                    NdArray param = auxParamMap.get(auxParams[j]);
                    if (param == null) {
                        throw new IllegalStateException("aux parameter not found: " + auxParams[j]);
                    }
                    param.copyTo(auxArray[j]);
                }
            }

            Pointer pointer = ref.getValue();
            NdArray[] out = JnaUtils.getExecutorOutputs(alloc, pointer);

            executors[i] =
                    new MxExecutor(alloc, pointer, argArray, auxArray, dataArray, out, gradArray);
        }
        return executors;
    }

    public String toJson() {
        return JnaUtils.symbolToJson(getHandle());
    }

    @Override
    public void close() {
        Pointer pointer = handle.getAndSet(null);
        if (pointer != null) {
            JnaUtils.freeSymbol(pointer);
        }
        if (alloc != null) {
            alloc.detach(this);
        }
    }
}
