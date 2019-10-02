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
import java.util.List;
import java.util.Map;
import org.apache.mxnet.jna.JnaUtils;
import org.apache.mxnet.nn.MxSymbolBlock;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import software.amazon.ai.Device;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.ndarray.types.Shape;
import software.amazon.ai.nn.Parameter;
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

    private List<Parameter> parameters;
    private MxNDArray[] allInputsNDArray;
    private PairList<String, Integer> dataIndices;
    private Map<String, Integer> dataIndicesMap;
    private Map<String, Integer> paramIndicesMap;

    private MxNDManager manager;

    /**
     * Create an instance of {@link CachedOp}.
     *
     * <p>It can be created by using {@link JnaUtils#createCachedOp(MxSymbolBlock, MxNDManager)}
     *
     * @param handle The C handle of the CachedOp
     * @param manager manager used to create NDArray
     * @param parameters parameter values
     * @param paramIndices parameter required by the model and their corresponding location
     * @param dataIndices input data names required by the model and their corresponding location
     */
    public CachedOp(
            Pointer handle,
            MxNDManager manager,
            List<Parameter> parameters,
            PairList<String, Integer> paramIndices,
            PairList<String, Integer> dataIndices) {
        super(handle);
        this.parameters = parameters;
        this.dataIndices = dataIndices;
        this.dataIndicesMap = dataIndices.toMap();
        this.paramIndicesMap = paramIndices.toMap();
        // holds all parameter and data NDArray values, final inputs to CachedOp
        this.manager = manager;
        manager.attach(this);
    }

    /**
     * Forwarding method of CachedOp.
     *
     * <p>All inputs will be assigned to the empty locations of the inputNDArray
     *
     * @param data input in {@link NDList} format
     * @return result {@link NDList}
     */
    public NDList forward(NDList data) {
        // reset the input data index at the beginning
        allInputsNDArray = new MxNDArray[dataIndices.size() + paramIndicesMap.size()];
        // check device of input
        Device device;
        if (data != null && data.size() > 0 && data.get(0) != null) {
            device = data.get(0).getDevice();
        } else {
            device = Device.defaultDevice();
        }
        // fill allInputsNDArray with parameter values on correct device
        for (Parameter parameter : parameters) {
            int index = paramIndicesMap.get(parameter.getName());
            MxNDArray arrayOnDevice = (MxNDArray) parameter.getArray(device);
            if (!arrayOnDevice.getDevice().equals(device)) {
                throw new IllegalStateException(
                        "Input device and parameter device does not match, if you are "
                                + "training on multi-gpu, make sure you passed numGpus in TrainingConfig and "
                                + "called DatasetUtils.split to split data on each GPU.");
            }
            allInputsNDArray[index] = arrayOnDevice;
        }

        // fill allInputsNDArray with data values
        int index = 0;
        for (Pair<String, NDArray> pair : data) {
            String inputName = pair.getKey();
            // if inputName not provided, value will follow the default order
            int idx = indexOf(inputName, index++);
            allInputsNDArray[idx] = (MxNDArray) pair.getValue();
        }
        // check the input, set as Shape(1) by default
        for (Pair<String, Integer> pair : dataIndices) {
            if (allInputsNDArray[pair.getValue()] == null) {
                // TODO: Do we need to set default to the input?
                String key = pair.getKey();
                if (!"prob_label".equals(key) && !"softmax_label".equals(key)) {
                    logger.warn("Input " + key + " not found, set NDArray to Shape(1) by default");
                }
                allInputsNDArray[pair.getValue()] = (MxNDArray) manager.create(new Shape(1));
            }
        }
        MxNDArray[] result = JnaUtils.cachedOpInvoke(manager, getHandle(), allInputsNDArray);
        return new NDList(result);
    }

    public MxNDArray[] getInputNDArray() {
        return allInputsNDArray;
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
            return dataIndices.valueAt(position);
        }

        Integer index = dataIndicesMap.get(inputName);
        if (index == null) {
            throw new IllegalArgumentException(
                    "Unknown input name: "
                            + inputName
                            + ", expected inputs: "
                            + dataIndicesMap.keySet().toString());
        }
        return index;
    }
}
