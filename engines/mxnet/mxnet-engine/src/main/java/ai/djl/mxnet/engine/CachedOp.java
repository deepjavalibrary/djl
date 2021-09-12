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
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Parameter;
import ai.djl.training.ParameterStore;
import ai.djl.util.NativeResource;
import ai.djl.util.Pair;
import ai.djl.util.PairList;
import com.sun.jna.Pointer;
import java.util.List;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * The {@code CachedOp} is an internal helper that provides the core functionality to execute a
 * {@link MxSymbolBlock}.
 *
 * <p>We don't recommended users interact with this class directly. Users should use {@link
 * ai.djl.inference.Predictor} instead. CachedOp is an operator that simplifies calling and
 * analyzing the input shape. It requires minimum input to do inference because most of the
 * information can be obtained from the model itself.
 */
public class CachedOp extends NativeResource<Pointer> {

    private static final Logger logger = LoggerFactory.getLogger(CachedOp.class);

    private List<Parameter> parameters;
    private PairList<String, Integer> dataIndices;
    private Map<String, Integer> dataIndicesMap;
    private List<Integer> paramIndices;
    private MxNDManager manager;

    /**
     * Creates an instance of {@link CachedOp}.
     *
     * <p>It can be created by using {@link JnaUtils#createCachedOp(MxSymbolBlock, MxNDManager,
     * boolean)}
     *
     * @param handle the C handle of the CachedOp
     * @param manager the manager used to create the NDArray
     * @param parameters the parameter values
     * @param paramIndices the parameters required by the model and their corresponding location
     * @param dataIndices the input data names required by the model and their corresponding
     *     location
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
        manager.attachInternal(getUid(), this);
    }

    /**
     * Assigns inputs to the empty locations of the input NDArray.
     *
     * @param parameterStore the parameterStore
     * @param data the input in {@link NDList} format
     * @param training true for a training forward pass
     * @return an {@link NDList}
     */
    public NDList forward(ParameterStore parameterStore, NDList data, boolean training) {
        // reset the input data index at the beginning
        MxNDArray[] allInputsNDArray = new MxNDArray[parameters.size()];
        // check device of input
        Device device = data.head().getDevice();
        // get the manager of the data

        MxNDManager inputManager = (MxNDManager) data.head().getManager();

        // fill allInputsNDArray with parameter values on correct device
        for (int index : paramIndices) {
            Parameter parameter = parameters.get(index);
            MxNDArray value = (MxNDArray) parameterStore.getValue(parameter, device, training);
            if (value == null) {
                throw new NullPointerException("Failed to find parameter from parameterStore");
            }
            allInputsNDArray[index] = value;
        }

        // fill allInputsNDArray with data values
        int index = 0;
        for (NDArray array : data) {
            String inputName = array.getName();
            // if inputName not provided, value will follow the default order
            int idx = indexOf(inputName, index++);
            allInputsNDArray[idx] = (MxNDArray) array;
        }

        // check the input, set as Shape(batchSize) by default
        for (Pair<String, Integer> pair : dataIndices) {
            if (allInputsNDArray[pair.getValue()] == null) {
                // TODO: Do we need to set default to the input?
                long batchSize = data.head().getShape().get(0);
                String key = pair.getKey();
                if (!"prob_label".equals(key) && !"softmax_label".equals(key)) {
                    logger.warn(
                            "Input "
                                    + key
                                    + " not found, set NDArray to Shape("
                                    + batchSize
                                    + ") by default");
                }
                allInputsNDArray[pair.getValue()] =
                        (MxNDArray) inputManager.create(new Shape(batchSize));
            }
        }
        MxNDArray[] result = JnaUtils.cachedOpInvoke(inputManager, getHandle(), allInputsNDArray);
        return new NDList(result);
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        Pointer pointer = handle.getAndSet(null);
        if (pointer != null) {
            manager.detachInternal(getUid());
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
