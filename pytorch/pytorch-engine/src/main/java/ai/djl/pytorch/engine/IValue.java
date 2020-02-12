/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

package ai.djl.pytorch.engine;

import ai.djl.ndarray.NDList;
import ai.djl.pytorch.jni.JniUtils;
import ai.djl.pytorch.jni.Pointer;
import java.util.Map;

/** IValue is a generic container for PyTorch inference. */
public class IValue {

    private Pointer handle;
    private PtNDManager manager;

    /**
     * Create an {@link IValue} container
     *
     * <p>Java representation of a TorchScript value, which is implemented as tagged union that can
     * be one of the supported types: https://pytorch.org/docs/stable/jit.html#types.
     *
     * @param manager {@link PtNDManager} that used to attach new NDArray
     * @param pointer the IValue handle pointer
     */
    public IValue(PtNDManager manager, Pointer pointer) {
        this.handle = pointer;
        this.manager = manager;
    }

    /**
     * Get the IValue handle.
     *
     * @return IValue {@link Pointer}
     */
    public Pointer getHandle() {
        return handle;
    }

    /**
     * Create an {@link IValue} container from NDArray.
     *
     * @param array {@link PtNDArray}
     * @return IValue container
     */
    public static IValue fromNDArray(PtNDArray array) {
        return JniUtils.createIValueFromNDArray(array);
    }

    /**
     * Get the {@link PtNDManager}.
     *
     * @return {@link PtNDManager}
     */
    public PtNDManager getManager() {
        return manager;
    }

    /**
     * Check IValue is a container of {@link PtNDArray}.
     *
     * @return result
     */
    public boolean isNDArray() {
        return JniUtils.iValueIsNDArray(this);
    }

    /**
     * Check IValue is a container of {@link NDList}.
     *
     * @return result
     */
    public boolean isNDList() {
        return JniUtils.iValueIsNDList(this);
    }

    /**
     * Check IValue is a container of IValue Array.
     *
     * @return result
     */
    public boolean isArray() {
        return JniUtils.iValueIsList(this);
    }

    /**
     * Check IValue is a container of IValue Map.
     *
     * @return result
     */
    public boolean isMap() {
        return JniUtils.iValueIsMap(this);
    }

    /**
     * Check IValue is a container of String.
     *
     * @return result
     */
    public boolean isString() {
        return JniUtils.iValueIsString(this);
    }

    /**
     * Extract IValue with a {@link PtNDArray} value.
     *
     * @return {@link ai.djl.ndarray.NDArray}
     */
    public PtNDArray toNDArray() {
        return JniUtils.iValueToNDArray(manager, this);
    }

    /**
     * Extract IValue to {@link NDList}.
     *
     * @return {@link NDList}
     */
    public NDList toNDList() {
        return JniUtils.iValueToNDList(this);
    }

    /**
     * Extract IValue to an IValue Array.
     *
     * @return IValue array
     */
    public IValue[] toIValueArray() {
        return JniUtils.iValueToIValueArray(this);
    }

    /**
     * Extract IValue to a Map.
     *
     * @return IValue Map
     */
    public Map<IValue, IValue> toIValueMap() {
        return JniUtils.iValueToIValueMap(this);
    }

    /**
     * Extract IValue to String.
     *
     * @return String
     */
    public String toIValueString() {
        return JniUtils.iValueToString(this);
    }
}
