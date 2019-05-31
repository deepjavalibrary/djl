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

import com.amazon.ai.ndarray.NDArray;
import com.amazon.ai.ndarray.NDList;
import com.amazon.ai.ndarray.types.DataDesc;
import com.amazon.ai.ndarray.types.Shape;
import com.amazon.ai.util.Pair;
import com.amazon.ai.util.PairList;
import com.sun.jna.Pointer;
import java.util.Map;
import org.apache.mxnet.jna.JnaUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class CachedOp extends NativeResource {

    private static final Logger logger = LoggerFactory.getLogger(CachedOp.class);

    private MxNDArray[] inputNDArray;
    private PairList<String, Integer> inputNames;
    private Map<String, Integer> inputNameMap;
    private MxNDFactory factory;

    public CachedOp(
            Pointer handle,
            MxNDFactory factory,
            MxNDArray[] inputNDArray,
            PairList<String, Integer> inputNames) {
        super(handle);
        this.inputNDArray = inputNDArray;
        this.inputNames = inputNames;
        inputNameMap = inputNames.toMap();
        this.factory = factory;
    }

    public NDList forward(NDList list) {
        int index = 0;
        for (Pair<String, NDArray> pair : list) {
            String inputName = pair.getKey();
            // if inputName not provided, value will follow the default order
            int position = indexOf(inputName, index++);
            // TODO: should we check context of input data?
            inputNDArray[position] = (MxNDArray) pair.getValue();
        }
        // check the input, set as Shape(1) by default
        for (Pair<String, Integer> pair : inputNames) {
            if (inputNDArray[pair.getValue()] == null) {
                // TODO: Do we need to set default to the input?
                logger.warn(
                        "Input "
                                + pair.getKey()
                                + " not found, set NDArray to Shape(1) by default");
                inputNDArray[pair.getValue()] = factory.create(new DataDesc(new Shape(1)));
            }
        }
        MxNDArray[] result = JnaUtils.cachedOpInvoke(factory, getHandle(), inputNDArray);
        return new NDList(result);
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        JnaUtils.freeCachedOp(getHandle());
    }

    private int indexOf(String inputName, int position) {
        if (inputName == null) {
            return inputNames.valueAt(position);
        }

        Integer index = inputNameMap.get(inputName);
        if (index == null) {
            throw new IllegalArgumentException(
                    "Unknown input name: "
                            + inputName
                            + ", expected inputs: "
                            + inputNameMap.keySet().toString());
        }
        return index;
    }
}
