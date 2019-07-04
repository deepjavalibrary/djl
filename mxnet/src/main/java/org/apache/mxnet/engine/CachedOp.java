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

import com.sun.jna.Pointer;
import java.util.Map;
import org.apache.mxnet.jna.JnaUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.ndarray.types.DataDesc;
import software.amazon.ai.ndarray.types.Shape;
import software.amazon.ai.util.Pair;
import software.amazon.ai.util.PairList;

/**
 * The {@code CachedOp} class provides the core functionality to execute a graph with MXNet.
 *
 * <p>Users are not recommended to interact with this class directly, use {@link
 * software.amazon.ai.inference.Predictor} instead. CachedOp is an operator that simplify the input
 * by self-analyzing the input shape such as the batch size. It require minimum input to do
 * inference since most of the information can be obtained from the model itself.
 */
public class CachedOp extends NativeResource {

    private static final Logger logger = LoggerFactory.getLogger(CachedOp.class);

    private MxNDArray[] inputNDArray;
    private PairList<String, Integer> inputNames;
    private Map<String, Integer> inputNameMap;
    private MxNDFactory factory;

    /**
     * Create an instance of {@link CachedOp}.
     *
     * <p>It can be created by using {@link JnaUtils#createCachedOp(MxModel, MxNDFactory)}
     *
     * @param handle The C handle of the CachedOp
     * @param factory factory used to create NDArray
     * @param inputNDArray The inputNDArray contains no inputs and all params
     * @param inputNames input names required by the model and their corresponding location
     */
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
        factory.attach(this);
    }

    /**
     * Forwarding method of CachedOp.
     *
     * <p>All inputs will be assigned to the empty locations of the inputNDArray
     *
     * @param list input in {@link NDList} format
     * @return result {@link NDList}
     */
    public NDList forward(NDList list) {
        // reset the input field at the beginning
        for (int location : inputNames.values()) {
            inputNDArray[location] = null;
        }
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
                String key = pair.getKey();
                if (!"prob_label".equals(key) && !"softmax_label".equals(key)) {
                    logger.warn("Input " + key + " not found, set NDArray to Shape(1) by default");
                }
                inputNDArray[pair.getValue()] = factory.create(new DataDesc(new Shape(1)));
            }
        }
        MxNDArray[] result = JnaUtils.cachedOpInvoke(factory, getHandle(), inputNDArray);
        return new NDList(result);
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        Pointer pointer = handle.getAndSet(null);
        if (pointer != null) {
            JnaUtils.freeCachedOp(pointer);
        }
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
