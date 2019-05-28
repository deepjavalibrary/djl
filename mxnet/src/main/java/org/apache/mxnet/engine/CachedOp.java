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

import com.amazon.ai.ndarray.NDList;
import com.amazon.ai.util.PairList;
import com.sun.jna.Pointer;
import com.sun.jna.ptr.PointerByReference;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import org.apache.mxnet.jna.JnaUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class CachedOp extends NativeResource {
    private static final Logger logger = LoggerFactory.getLogger(CachedOp.class);

    private MxNDArray[] inputNDArray;
    private String[] inputNames;
    private int[] inputLocation;
    private MxNDFactory factory;

    public CachedOp(
            Pointer handle,
            MxNDFactory factory,
            MxNDArray[] inputNDArray,
            String[] inputNames,
            int[] inputLocation) {
        super(handle);
        this.inputNDArray = inputNDArray;
        this.inputNames = inputNames;
        this.inputLocation = inputLocation;
        this.factory = factory;
    }

    public static CachedOp loadModel(MxNDFactory factory, String prefix, int epoch) {
        Symbol symbol = Symbol.load(factory, prefix + "-symbol.json");
        String paramFile = String.format("%s-%04d.params", prefix, epoch);
        // Getting all names and param names
        PointerByReference namesRef = new PointerByReference();
        Pointer[] handles = JnaUtils.loadNdArray(paramFile, namesRef);
        String[] paramNamesRaw = namesRef.getValue().getStringArray(0, handles.length);
        String[] allNames = symbol.getAllNames();
        int allNameLength = allNames.length;
        int paramLength = paramNamesRaw.length;
        Map<String, Pointer> paramNameToHandle = new HashMap<>();
        for (int i = 0; i < paramLength; i++) {
            String paramName = paramNamesRaw[i].split(":")[1];
            paramNameToHandle.put(paramName, handles[i]);
        }
        // prepare the cachedOp element
        String[] inputNames = new String[allNameLength - paramLength];
        MxNDArray[] inputNDArray = new MxNDArray[allNameLength];
        // indices of data input and param
        int[] dataIndices = new int[allNameLength - paramLength];
        int[] paramIndices = new int[paramLength];
        // Start forming input array and param indices
        int dataLoc = 0;
        int paramLoc = 0;
        for (int idx = 0; idx < allNameLength; ++idx) {
            if (paramNameToHandle.containsKey(allNames[idx])) {
                paramIndices[paramLoc] = idx;
                Pointer handle = paramNameToHandle.get(allNames[idx]);
                // TODO: Change the DataType to non-determined
                inputNDArray[idx] = factory.create(handle).asInContext(factory.getContext(), true);
                paramLoc++;
            } else {
                inputNames[dataLoc] = allNames[idx];
                dataIndices[dataLoc] = idx;
                dataLoc++;
            }
        }
        // Creating CachedOp
        String[] keys =
                new String[] {"data_indices", "param_indices", "static_alloc", "static_shape"};
        String[] values =
                new String[] {
                    Arrays.toString(dataIndices), Arrays.toString(paramIndices), "1", "1"
                };
        if (logger.isDebugEnabled()) {
            logger.debug("keys: {}", Arrays.toString(keys));
            logger.debug("indices: {}", Arrays.toString(values));
        }
        PairList<String, String> flags = new PairList<>(keys, values);
        Pointer handle = JnaUtils.createCachedOp(symbol, flags);

        return new CachedOp(handle, factory, inputNDArray, inputNames, dataIndices);
    }

    public String[] getInputNames() {
        return this.inputNames;
    }

    public NDList forward(NDList list) {
        if (list.size() != inputNames.length) {
            throw new IllegalArgumentException(
                    "Input size mismatch! Expected Inputs: " + Arrays.toString(getInputNames()));
        }
        int[] locations = inputLocation;
        for (int i = 0; i < list.size(); i++) {
            inputNDArray[locations[i]] = (MxNDArray) list.get(i);
        }
        MxNDArray[] result = JnaUtils.cachedOpInvoke(factory, getHandle(), inputNDArray);
        return new NDList(result);
    }

    @Override
    public void close() {
        JnaUtils.freeCachedOp(getHandle());
    }
}
