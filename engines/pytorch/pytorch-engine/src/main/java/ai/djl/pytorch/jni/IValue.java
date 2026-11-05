/*
 * Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.pytorch.jni;

import ai.djl.Device;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.pytorch.engine.PtDeviceType;
import ai.djl.pytorch.engine.PtNDArray;
import ai.djl.pytorch.engine.PtNDManager;
import ai.djl.util.NativeResource;

import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

/**
 * A class represent a PyTorch {@code IValue} data.
 *
 * <p>DJL doesn't support creating nested IValue.
 */
public class IValue extends NativeResource<Long> {

    // keeps the source buffers reachable while native tensors point into them
    private Object bufferRef;

    IValue(long handle) {
        super(handle);
    }

    /**
     * Returns the type of the IValue.
     *
     * @return the type of the IValue
     */
    public String getType() {
        return PyTorchLibrary.LIB.iValueGetType(getHandle());
    }

    /**
     * Returns if the IValue is a {@code Tensor} type.
     *
     * @return if the IValue is a Tensor type
     */
    public boolean isTensor() {
        return PyTorchLibrary.LIB.iValueIsTensor(getHandle());
    }

    /**
     * Returns if the IValue is a {@code boolean} type.
     *
     * @return if the IValue is a boolean type
     */
    public boolean isBoolean() {
        return PyTorchLibrary.LIB.iValueIsBool(getHandle());
    }

    /**
     * Returns if the IValue is a {@code long} type.
     *
     * @return if the IValue is a long type
     */
    public boolean isLong() {
        return PyTorchLibrary.LIB.iValueIsLong(getHandle());
    }

    /**
     * Returns if the IValue is a {@code double} type.
     *
     * @return if the IValue is a double type
     */
    public boolean isDouble() {
        return PyTorchLibrary.LIB.iValueIsDouble(getHandle());
    }

    /**
     * Returns if the IValue is a {@code String} type.
     *
     * @return if the IValue is a String type
     */
    public boolean isString() {
        return PyTorchLibrary.LIB.iValueIsString(getHandle());
    }

    /**
     * Returns if the IValue is a {@code boolean[]} type.
     *
     * @return if the IValue is a boolean[] type
     */
    public boolean isBooleanList() {
        return PyTorchLibrary.LIB.iValueIsBoolList(getHandle());
    }

    /**
     * Returns if the IValue is a {@code long[]} type.
     *
     * @return if the IValue is a long[] type
     */
    public boolean isLongList() {
        return PyTorchLibrary.LIB.iValueIsLongList(getHandle());
    }

    /**
     * Returns if the IValue is a {@code double[]} type.
     *
     * @return if the IValue is a double[] type
     */
    public boolean isDoubleList() {
        return PyTorchLibrary.LIB.iValueIsDoubleList(getHandle());
    }

    /**
     * Returns if the IValue is a {@code IValue[]} type.
     *
     * <p>The elements in the array must have the same type.
     *
     * @return if the IValue is a IValue[] type
     */
    public boolean isTensorList() {
        return PyTorchLibrary.LIB.iValueIsTensorList(getHandle());
    }

    /**
     * Returns if the IValue is a {@code IValue[]} type.
     *
     * <p>The elements in the array must have the same type.
     *
     * @return if the IValue is a IValue[] type
     */
    public boolean isList() {
        return PyTorchLibrary.LIB.iValueIsList(getHandle());
    }

    /**
     * Returns if the IValue is a {@code Map&lt;String, V&gt;} type.
     *
     * @return if the IValue is a Map&lt;String, V&gt; type
     */
    public boolean isMap() {
        return PyTorchLibrary.LIB.iValueIsMap(getHandle());
    }

    /**
     * Returns if the IValue is a tuple type.
     *
     * @return if the IValue is a tuple type
     */
    public boolean isTuple() {
        return PyTorchLibrary.LIB.iValueIsTuple(getHandle());
    }

    /**
     * Creates a new {@code IValue} of type {@code PtNDArray}.
     *
     * @param array the NDArray
     * @return a new {@code IValue} of type {@code PtNDArray}
     */
    public static IValue from(PtNDArray array) {
        if (array.getDataType() == DataType.STRING) {
            Shape shape = array.getShape();
            String[] strs = array.toStringArray();
            if (shape.isScalar()) {
                return from(strs[0]);
            }
            IValue[] list = new IValue[strs.length];
            PtNDManager manager = array.getManager();
            for (int i = 0; i < strs.length; i++) {
                IValue ivalue = from(strs[i]);
                manager.attachUncappedInternal(ivalue.getUid(), ivalue);
                list[i] = ivalue;
            }
            return listFrom(list);
        }
        return new IValue(PyTorchLibrary.LIB.iValueFromTensor(array.getHandle()));
    }

    /**
     * Creates a new {@code IValue} of type {@code boolean}.
     *
     * @param value the boolean value
     * @return a new {@code IValue} of type {@code boolean}
     */
    public static IValue from(boolean value) {
        return new IValue(PyTorchLibrary.LIB.iValueFromBool(value));
    }

    /**
     * Creates a new {@code IValue} of type {@code long}.
     *
     * @param value the long value
     * @return a new {@code IValue} of type {@code long}
     */
    public static IValue from(long value) {
        return new IValue(PyTorchLibrary.LIB.iValueFromLong(value));
    }

    /**
     * Creates a new {@code IValue} of type {@code double}.
     *
     * @param value the double value
     * @return a new {@code IValue} of type {@code double}
     */
    public static IValue from(double value) {
        return new IValue(PyTorchLibrary.LIB.iValueFromDouble(value));
    }

    /**
     * Creates a new {@code IValue} of type {@code String}.
     *
     * @param value the String value
     * @return a new {@code IValue} of type {@code String}
     */
    public static IValue from(String value) {
        return new IValue(PyTorchLibrary.LIB.iValueFromString(value));
    }

    /**
     * Creates a new {@code IValue} of type {@code boolean[]}.
     *
     * @param list the boolean[] value
     * @return a new {@code IValue} of type {@code boolean[]}
     */
    public static IValue listFrom(boolean... list) {
        return new IValue(PyTorchLibrary.LIB.iValueFromBoolList(list));
    }

    /**
     * Creates a new {@code IValue} of type {@code long[]}.
     *
     * @param list the long[] value
     * @return a new {@code IValue} of type {@code long[]}
     */
    public static IValue listFrom(long... list) {
        return new IValue(PyTorchLibrary.LIB.iValueFromLongList(list));
    }

    /**
     * Creates a new {@code IValue} of type {@code double[]}.
     *
     * @param list the double[] value
     * @return a new {@code IValue} of type {@code double[]}
     */
    public static IValue listFrom(double... list) {
        return new IValue(PyTorchLibrary.LIB.iValueFromDoubleList(list));
    }

    /**
     * Creates a new {@code IValue} of type {@code NDArray[]}.
     *
     * @param list the NDArray[] value
     * @return a new {@code IValue} of type {@code NDArray[]}
     */
    public static IValue listFrom(PtNDArray... list) {
        long[] tensors = Arrays.stream(list).mapToLong(PtNDArray::getHandle).toArray();
        return new IValue(PyTorchLibrary.LIB.iValueFromTensorList(tensors));
    }

    /**
     * Creates a new {@code IValue} of type {@code NDArray[]}.
     *
     * @param list the NDArray[] value
     * @return a new {@code IValue} of type {@code NDArray[]}
     */
    public static IValue listFrom(IValue... list) {
        if (list.length == 0) {
            throw new IllegalArgumentException("Empty IValue list is not supported.");
        }
        long[] tensors = Arrays.stream(list).mapToLong(IValue::getHandle).toArray();
        return new IValue(PyTorchLibrary.LIB.iValueFromList(tensors));
    }

    /**
     * Creates a new {@code IValue} of type {@code NDArray[]}.
     *
     * @param list the NDArray[] value
     * @return a new {@code IValue} of type {@code NDArray[]}
     */
    public static IValue tupleFrom(IValue... list) {
        long[] tensors = Arrays.stream(list).mapToLong(IValue::getHandle).toArray();
        return new IValue(PyTorchLibrary.LIB.iValueFromTuple(tensors));
    }

    /**
     * Creates a new {@code IValue} of type {@code Map[String, PtNDArray]}.
     *
     * @param map the Map[String, IValue] value
     * @return a new {@code IValue} of type {@code Map[String, PtNDArray]}
     */
    public static IValue stringMapFrom(Map<String, PtNDArray> map) {
        String[] keys = new String[map.size()];
        long[] handles = new long[map.size()];
        int i = 0;
        for (Map.Entry<String, PtNDArray> entry : map.entrySet()) {
            keys[i] = entry.getKey();
            handles[i] = entry.getValue().getHandle();
            ++i;
        }
        return new IValue(PyTorchLibrary.LIB.iValueFromStringMap(keys, handles));
    }

    /**
     * Creates a new {@code IValue} of type {@code Dict[str, Tensor]} directly from direct buffers
     * in a single JNI call. On CPU nothing is copied: the tensors point into the caller's buffers.
     * The buffers are kept reachable until this {@code IValue} is closed.
     *
     * <p>Values derived from the dictionary ({@link #toIValueMap()}, {@link
     * #toTensor(PtNDManager)}, model outputs that alias an input, results of {@code
     * PtSymbolBlock.forward(IValue...)}) share that memory and do NOT keep it alive. They must not
     * be used after this {@code IValue} (or the carrier owning it) is closed, unless the caller
     * keeps the buffers reachable.
     *
     * <p>{@link ByteBuffer#slice()} returns a BIG_ENDIAN view regardless of its parent, so callers
     * must call {@code .order(ByteOrder.nativeOrder())} on each slice before writing, otherwise the
     * values are silently wrong.
     *
     * <p>The {@code position} and {@code limit} of the buffers are ignored: data is read from the
     * buffer's start address (the slice start for slices), and the capacity must cover the shape.
     *
     * @param keys the dictionary keys, must be non-null and unique
     * @param buffers the direct buffers holding the tensor data
     * @param shapes the shape of each tensor
     * @param dtypes the data type of each tensor
     * @return a new {@code IValue} of type {@code Dict[str, Tensor]}
     * @throws IllegalArgumentException if the arguments are invalid
     * @throws ai.djl.engine.EngineException if the native check fails, e.g. a buffer is too small
     */
    public static IValue stringMapFrom(
            String[] keys, ByteBuffer[] buffers, long[][] shapes, DataType[] dtypes) {
        int n = keys.length;
        if (n == 0) {
            throw new IllegalArgumentException("At least one entry is required");
        }
        if (buffers.length != n || shapes.length != n || dtypes.length != n) {
            throw new IllegalArgumentException("keys, buffers, shapes and dtypes must match");
        }
        Set<String> seen = new HashSet<>();
        int[] dtypeOrdinals = new int[n];
        for (int i = 0; i < n; ++i) {
            if (keys[i] == null) {
                throw new IllegalArgumentException("Null key at index " + i);
            }
            if (!seen.add(keys[i])) {
                throw new IllegalArgumentException("Duplicate key: " + keys[i]);
            }
            if (buffers[i] == null || !buffers[i].isDirect()) {
                throw new IllegalArgumentException("A direct ByteBuffer is required: " + keys[i]);
            }
            if (shapes[i] == null) {
                throw new IllegalArgumentException("Null shape for key: " + keys[i]);
            }
            for (long dim : shapes[i]) {
                if (dim < 0) {
                    throw new IllegalArgumentException("Invalid shape for key: " + keys[i]);
                }
            }
            if (dtypes[i] == DataType.COMPLEX64) {
                throw new IllegalArgumentException(
                        "COMPLEX64 is not supported yet (see #3927) for key: " + keys[i]);
            }
            if (dtypes[i] == null || dtypes[i].ordinal() > DataType.BOOLEAN.ordinal()) {
                throw new IllegalArgumentException(
                        "Unsupported data type " + dtypes[i] + " for key: " + keys[i]);
            }
            dtypeOrdinals[i] = dtypes[i].ordinal();
        }
        int[] device = {PtDeviceType.toDeviceType(Device.cpu()), -1};
        long handle =
                PyTorchLibrary.LIB.iValueFromStringMapBlobs(
                        keys, buffers, shapes, dtypeOrdinals, device);
        IValue ret = new IValue(handle);
        ret.bufferRef = buffers.clone();
        return ret;
    }

    /**
     * Creates a new {@code IValue} of type {@code Map[String, IValue]}.
     *
     * @param map the Map[String, IValue] value
     * @return a new {@code IValue} of type {@code Map[String, IValue]}
     */
    public static IValue stringIValueMapFrom(Map<String, IValue> map) {
        String[] keys = new String[map.size()];
        long[] handles = new long[map.size()];
        int i = 0;
        for (Map.Entry<String, IValue> entry : map.entrySet()) {
            keys[i] = entry.getKey();
            handles[i] = entry.getValue().getHandle();
            ++i;
        }
        return new IValue(PyTorchLibrary.LIB.iValueFromStringIValueMap(keys, handles));
    }

    /**
     * Returns the {@code boolean} value of this IValue.
     *
     * @return the boolean value of this IValue
     */
    public boolean toBoolean() {
        return PyTorchLibrary.LIB.iValueToBool(getHandle());
    }

    /**
     * Returns the {@code long} value of this IValue.
     *
     * @return the long value of this IValue
     */
    public long toLong() {
        return PyTorchLibrary.LIB.iValueToLong(getHandle());
    }

    /**
     * Returns the {@code double} value of this IValue.
     *
     * @return the double value of this IValue
     */
    public double toDouble() {
        return PyTorchLibrary.LIB.iValueToDouble(getHandle());
    }

    /**
     * Returns the {@code String} value of this IValue.
     *
     * @return the String value of this IValue
     */
    public String toStringValue() {
        return PyTorchLibrary.LIB.iValueToString(getHandle());
    }

    /**
     * Returns the {@code boolean[]} value of this IValue.
     *
     * @return the boolean[] value of this IValue
     */
    public boolean[] toBooleanArray() {
        return PyTorchLibrary.LIB.iValueToBoolList(getHandle());
    }

    /**
     * Returns the {@code long[]} value of this IValue.
     *
     * @return the long[] value of this IValue
     */
    public long[] toLongArray() {
        return PyTorchLibrary.LIB.iValueToLongList(getHandle());
    }

    /**
     * Returns the {@code double[]} value of this IValue.
     *
     * @return the double[] value of this IValue
     */
    public double[] toDoubleArray() {
        return PyTorchLibrary.LIB.iValueToDoubleList(getHandle());
    }

    /**
     * Returns the {@code NDArray} value of this IValue.
     *
     * @param manager the {@code NDManager} to create the NDArray
     * @return the NDArray value of this IValue
     */
    public PtNDArray toTensor(PtNDManager manager) {
        return new PtNDArray(manager, PyTorchLibrary.LIB.iValueToTensor(getHandle()));
    }

    /**
     * Returns the {@code NDArray[]} value of this IValue.
     *
     * @param manager the NDManager to create NDArray
     * @return the NDArray[] value of this IValue
     */
    public PtNDArray[] toTensorArray(PtNDManager manager) {
        long[] handles = PyTorchLibrary.LIB.iValueToTensorList(getHandle());
        PtNDArray[] ret = new PtNDArray[handles.length];
        for (int i = 0; i < ret.length; ++i) {
            ret[i] = new PtNDArray(manager, handles[i]);
        }
        return ret;
    }

    /**
     * Returns the {@code IValue[]} value of this IValue list.
     *
     * @return the IValue[] value of this IValue list
     */
    public IValue[] toIValueArray() {
        long[] handles = PyTorchLibrary.LIB.iValueToIValueList(getHandle());
        IValue[] ret = new IValue[handles.length];
        for (int i = 0; i < ret.length; ++i) {
            ret[i] = new IValue(handles[i]);
        }
        return ret;
    }

    /**
     * Returns the {@code Map&lt;String, IValue&gt;} value of this IValue.
     *
     * @return the Map&lt;String, IValue&gt; value of this IValue
     */
    public Map<String, IValue> toIValueMap() {
        long[] handles = PyTorchLibrary.LIB.iValueToMap(getHandle());
        Map<String, IValue> map = new ConcurrentHashMap<>();
        for (int i = 0; i < handles.length; i += 2) {
            IValue key = new IValue(handles[i]);
            map.put(key.toStringValue(), new IValue(handles[i + 1]));
            key.close();
        }
        return map;
    }

    /**
     * Returns the {@code Map&lt;String, IValue&gt;} value of this IValue.
     *
     * @return the Map&lt;String, IValue&gt; value of this IValue
     */
    public IValue[] toIValueTuple() {
        long[] handles = PyTorchLibrary.LIB.iValueToIValueTuple(getHandle());
        IValue[] ret = new IValue[handles.length];
        for (int i = 0; i < ret.length; ++i) {
            ret[i] = new IValue(handles[i]);
        }
        return ret;
    }

    /**
     * Returns the {@code NDList} value of this IValue.
     *
     * @param manager the NDManager to create NDArray
     * @return the {@code NDList} value of this IValue
     */
    public NDList toNDList(PtNDManager manager) {
        if (isTensor()) {
            return new NDList(toTensor(manager));
        } else if (isTensorList()) {
            return new NDList(toTensorArray(manager));
        } else if (isMap()) {
            // Only allows one level <String, NDArray> type of map
            NDList list = new NDList();
            Map<String, IValue> map = toIValueMap();
            for (Map.Entry<String, IValue> entry : map.entrySet()) {
                IValue iv = entry.getValue();
                if (!iv.isTensor()) {
                    throw new UnsupportedOperationException("Only one level of map is supported.");
                }
                PtNDArray value = entry.getValue().toTensor(manager);
                value.setName(entry.getKey());
                list.add(value);
                iv.close();
            }
            return list;
        } else if (isList()) {
            NDList list = new NDList();
            for (IValue ivalue : toIValueArray()) {
                list.addAll(ivalue.toNDList(manager));
                ivalue.close();
            }
            return list;
        } else if (isTuple()) {
            NDList list = new NDList();
            for (IValue ivalue : toIValueTuple()) {
                list.addAll(ivalue.toNDList(manager));
                ivalue.close();
            }
            return list;
        } else if (isString()) {
            return new NDList(manager.create(toStringValue()));
        }
        throw new UnsupportedOperationException("Unsupported IValue type.");
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        Long pointer = handle.getAndSet(null);
        if (pointer != null) {
            PyTorchLibrary.LIB.torchDeleteIValue(pointer);
        }
        bufferRef = null;
    }
}
