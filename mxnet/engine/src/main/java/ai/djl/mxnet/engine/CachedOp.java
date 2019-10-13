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
import ai.djl.inference.Predictor;
import ai.djl.mxnet.jna.JnaUtils;
import ai.djl.mxnet.nn.MxSymbolBlock;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Parameter;
import ai.djl.training.ParameterStore;
import ai.djl.util.Pair;
import ai.djl.util.PairList;
import com.sun.jna.Pointer;
import java.util.List;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * The {@code CachedOp} class provides the core functionality to execute a graph with MXNet.
 *
 * <p>Users are not recommended to interact with this class directly, use {@link Predictor} instead.
 * CachedOp is an operator that simplify the input by self-analyzing the input shape such as the
 * batch size. It require minimum input to do inference since most of the information can be
 * obtained from the model itself.
 */
public class CachedOp extends NativeResource {

    private static final Logger logger = LoggerFactory.getLogger(CachedOp.class);

    private List<Parameter> parameters;
    private MxNDArray[] allInputsNDArray;
    private PairList<String, Integer> dataIndices;
    private Map<String, Integer> dataIndicesMap;
    private List<Integer> paramIndices;

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
            List<Integer> paramIndices,
            PairList<String, Integer> dataIndices) {
        super(handle);
        this.parameters = parameters;
        this.dataIndices = dataIndices;
        this.paramIndices = paramIndices;
        this.dataIndicesMap = dataIndices.toMap();
        // holds all parameter and data NDArray values, final inputs to CachedOp
        this.manager = manager;
        manager.attach(getUid(), this);
    }

    /**
     * Forwarding method of CachedOp.
     *
     * <p>All inputs will be assigned to the empty locations of the inputNDArray
     *
     * @param parameterStore ParameterStore
     * @param data input in {@link NDList} format
     * @return result {@link NDList}
     */
    public NDList forward(ParameterStore parameterStore, NDList data) {
        // reset the input data index at the beginning
        allInputsNDArray = new MxNDArray[parameters.size()];
        // check device of input
        Device device = data.head().getDevice();

        // fill allInputsNDArray with parameter values on correct device
        for (int index : paramIndices) {
            Parameter parameter = parameters.get(index);
            allInputsNDArray[index] = (MxNDArray) parameterStore.getValue(parameter, device);
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

    /**
     * For unit test only.
     *
     * @return array of NDArray
     */
    MxNDArray[] getInputNDArray() {
        return allInputsNDArray;
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        Pointer pointer = handle.getAndSet(null);
        if (pointer != null) {
            manager.detach(getUid());
            JnaUtils.freeCachedOp(pointer);
            manager = null;
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
