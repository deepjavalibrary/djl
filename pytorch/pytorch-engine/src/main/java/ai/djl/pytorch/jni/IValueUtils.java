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

package ai.djl.pytorch.jni;

import ai.djl.ndarray.NDList;
import ai.djl.pytorch.engine.PtNDArray;
import ai.djl.pytorch.engine.PtNDManager;
import ai.djl.pytorch.engine.PtSymbolBlock;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/** IValueUtils is utility class to deal with IValue in PyTorch. */
public final class IValueUtils {

    private IValueUtils() {}

    /**
     * Create IValue Pointer from NDArray.
     *
     * @param array {@link PtNDArray}
     * @return IValue Pointer
     */
    public static Pointer toIValuePointer(PtNDArray array) {
        return PyTorchLibrary.LIB.iValueCreateFromTensor(array.getHandle());
    }

    /**
     * Check IValue is a container of {@link PtNDArray}.
     *
     * @param iValueHandle IValue {@link Pointer}
     * @return result
     */
    public static boolean isNDArray(Pointer iValueHandle) {
        return PyTorchLibrary.LIB.iValueIsTensor(iValueHandle);
    }

    /**
     * Check IValue is a container of {@link NDList}.
     *
     * @param iValueHandle IValue {@link Pointer}
     * @return result
     */
    public static boolean isNDList(Pointer iValueHandle) {
        return PyTorchLibrary.LIB.iValueIsTensorList(iValueHandle);
    }

    /**
     * Check IValue is a container of IValue List.
     *
     * @param iValueHandle IValue {@link Pointer}
     * @return result
     */
    public static boolean isList(Pointer iValueHandle) {
        return PyTorchLibrary.LIB.iValueIsList(iValueHandle);
    }

    /**
     * Check IValue is a container of IValue Tuple.
     *
     * @param iValueHandle IValue {@link Pointer}
     * @return result
     */
    public static boolean isTuple(Pointer iValueHandle) {
        return PyTorchLibrary.LIB.iValueIsTuple(iValueHandle);
    }

    /**
     * Check IValue is a container of IValue Map.
     *
     * @param iValueHandle IValue {@link Pointer}
     * @return result
     */
    public static boolean isMap(Pointer iValueHandle) {
        return PyTorchLibrary.LIB.iValueIsMap(iValueHandle);
    }

    /**
     * Check IValue is a container of String.
     *
     * @param iValueHandle IValue {@link Pointer}
     * @return result
     */
    public static boolean isString(Pointer iValueHandle) {
        return PyTorchLibrary.LIB.iValueIsString(iValueHandle);
    }

    /**
     * Extract IValue with a {@link PtNDArray} value.
     *
     * @param iValueHandle IValue {@link Pointer}
     * @param manager {@link PtNDManager} that creates {@link PtNDArray}
     * @return {@link ai.djl.ndarray.NDArray}
     */
    public static PtNDArray toNDArray(Pointer iValueHandle, PtNDManager manager) {
        Pointer ndHandle = PyTorchLibrary.LIB.iValueToTensor(iValueHandle);
        return manager.create(ndHandle);
    }

    /**
     * Extract IValue to {@link NDList}.
     *
     * @param iValueHandle IValue {@link Pointer}
     * @param manager {@link PtNDManager} that creates {@link PtNDArray}
     * @return {@link NDList}
     */
    public static NDList toNDList(Pointer iValueHandle, PtNDManager manager) {
        Pointer[] ndHandles = PyTorchLibrary.LIB.iValueToTensorList(iValueHandle);
        NDList list = new NDList();
        for (Pointer handle : ndHandles) {
            list.add(manager.create(handle));
        }
        return list;
    }

    /**
     * Extract IValue to String.
     *
     * @param iValueHandle IValue {@link Pointer}
     * @return String
     */
    public static String toString(Pointer iValueHandle) {
        return PyTorchLibrary.LIB.iValueToString(iValueHandle);
    }

    /**
     * Extract IValue to an IValue Array.
     *
     * @param iValueHandle IValue {@link Pointer}
     * @return IValue array
     */
    public static Pointer[] toIValueArray(Pointer iValueHandle) {
        if (isTuple(iValueHandle)) {
            return PyTorchLibrary.LIB.iValueToListFromTuple(iValueHandle);
        }
        return PyTorchLibrary.LIB.iValueToList(iValueHandle);
    }

    /**
     * Extract IValue to a Map.
     *
     * @param iValueHandle IValue {@link Pointer}
     * @return IValue Map
     */
    public static Map<Pointer, Pointer> toIValueMap(Pointer iValueHandle) {
        Pointer[] iValueHandles = PyTorchLibrary.LIB.iValueToMap(iValueHandle);
        Map<Pointer, Pointer> map = new ConcurrentHashMap<>();
        for (int i = 0; i < iValueHandles.length; i += 2) {
            map.put(iValueHandles[i], iValueHandles[i + 1]);
        }
        return map;
    }

    private static NDList forwardHelper(Pointer iValueHandle, PtNDManager manager) {
        NDList list = new NDList();
        if (isNDArray(iValueHandle)) {
            list.add(toNDArray(iValueHandle, manager));
        } else if (isNDList(iValueHandle)) {
            list.addAll(toNDList(iValueHandle, manager));
        } else if (isList(iValueHandle) || isTuple(iValueHandle)) {
            for (Pointer handle : toIValueArray(iValueHandle)) {
                list.addAll(forwardHelper(handle, manager));
            }
        } else if (isMap(iValueHandle)) {
            // Only allows <String, NDArray> type of map
            Map<Pointer, Pointer> map = toIValueMap(iValueHandle);
            for (Map.Entry<Pointer, Pointer> entry : map.entrySet()) {
                String name = toString(entry.getKey());
                // free the IValue handle
                PyTorchLibrary.LIB.torchDeleteIValue(entry.getKey());
                PtNDArray value = toNDArray(entry.getValue(), manager);
                // free the IValue handle
                PyTorchLibrary.LIB.torchDeleteIValue(entry.getValue());
                value.setName(name);
                list.add(value);
            }
        } else {
            // free the IValue handle
            PyTorchLibrary.LIB.torchDeleteIValue(iValueHandle);
            throw new UnsupportedOperationException("Unsupported IValue type");
        }
        // free the IValue handle
        PyTorchLibrary.LIB.torchDeleteIValue(iValueHandle);
        return list;
    }

    /**
     * Run the forward of PyTorch module.
     *
     * @param block the block that contains PyTorch module
     * @param inputs input {@link NDList}
     * @param isTrain is running on training mode
     * @return result {@link NDList}
     */
    public static NDList forward(PtSymbolBlock block, NDList inputs, boolean isTrain) {
        Pointer[] arrayHandles =
                inputs.stream()
                        .map(input -> ((PtNDArray) input).getHandle())
                        .toArray(Pointer[]::new);

        Pointer result = PyTorchLibrary.LIB.moduleForward(block.getHandle(), arrayHandles, isTrain);
        PtNDManager manager = (PtNDManager) inputs.get(0).getManager();
        return forwardHelper(result, manager);
    }
}
