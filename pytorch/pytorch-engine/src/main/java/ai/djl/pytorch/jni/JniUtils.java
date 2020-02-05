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
package ai.djl.pytorch.jni;

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.ndarray.types.SparseFormat;
import ai.djl.pytorch.engine.PtDeviceType;
import ai.djl.pytorch.engine.PtNDArray;
import ai.djl.pytorch.engine.PtNDManager;
import ai.djl.pytorch.engine.PtSymbolBlock;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;

public class JniUtils {
    private JniUtils() {}

    private static int layoutMapper(SparseFormat fmt) {
        if (fmt == SparseFormat.DENSE) {
            return 0;
        } else if (fmt == SparseFormat.UNDEFINED) {
            throw new UnsupportedOperationException("Type not supported");
        } else {
            return 1;
        }
    }

    public static String libraryVersion() {
        return String.valueOf(PyTorchLibrary.LIB.torchVersion());
    }

    // TODO: Unchecked Datatype and device mapping
    public static PtNDArray CreateNdFromByteBuffer(
            PtNDManager manager,
            ByteBuffer data,
            Shape shape,
            DataType dType,
            SparseFormat fmt,
            Device device) {
        int layoutVal = layoutMapper(fmt);
        return new PtNDArray(
                manager,
                PyTorchLibrary.LIB.torchFromBlob(
                        data,
                        shape.getShape(),
                        dType.ordinal(),
                        layoutVal,
                        new int[] {device.getDeviceId()},
                        false));
    }

    public static PtNDArray createEmptyNdArray(
            PtNDManager manager, Shape shape, DataType dType, Device device, SparseFormat fmt) {
        int layoutVal = layoutMapper(fmt);
        // TODO: set default type of require gradient
        return new PtNDArray(
                manager,
                PyTorchLibrary.LIB.torchEmpty(
                        shape.getShape(),
                        dType.ordinal(),
                        layoutVal,
                        new int[] {device.getDeviceId()},
                        false));
    }

    public static PtNDArray createZerosNdArray(
            PtNDManager manager, Shape shape, DataType dType, Device device, SparseFormat fmt) {
        int layoutVal = layoutMapper(fmt);
        // TODO: set default type of require gradient
        return new PtNDArray(
                manager,
                PyTorchLibrary.LIB.torchZeros(
                        shape.getShape(),
                        dType.ordinal(),
                        layoutVal,
                        new int[] {device.getDeviceId()},
                        false));
    }

    public static PtNDArray createOnesNdArray(
            PtNDManager manager, Shape shape, DataType dType, Device device, SparseFormat fmt) {
        int layoutVal = layoutMapper(fmt);
        // TODO: set default type of require gradient
        return new PtNDArray(
                manager,
                PyTorchLibrary.LIB.torchOnes(
                        shape.getShape(),
                        dType.ordinal(),
                        layoutVal,
                        new int[] {device.getDeviceId()},
                        false));
    }

    public static PtNDArray reshape(PtNDArray ndArray, long[] shape) {
        return new PtNDArray(
                (PtNDManager) ndArray.getManager(),
                PyTorchLibrary.LIB.torchReshape(ndArray.getHandle(), shape));
    }

    public static PtNDArray softmax(PtNDArray ndArray, int dim, DataType dTpe) {
        return new PtNDArray(
                (PtNDManager) ndArray.getManager(),
                PyTorchLibrary.LIB.torchSoftmax(ndArray.getHandle(), dim, dTpe.ordinal()));
    }

    public static PtNDArray argMax(PtNDArray ndArray) {
        return new PtNDArray(
                (PtNDManager) ndArray.getManager(),
                PyTorchLibrary.LIB.torchArgMax(ndArray.getHandle()));
    }

    public static PtNDArray argMax(PtNDArray ndArray, int dim, boolean keepDim) {
        return new PtNDArray(
                (PtNDManager) ndArray.getManager(),
                PyTorchLibrary.LIB.torchArgMax(ndArray.getHandle(), dim, keepDim));
    }

    public static NDList split(PtNDArray ndArray, long size, long axis) {
        Pointer[] ndPtrs = PyTorchLibrary.LIB.torchSplit(ndArray.getHandle(), size, axis);
        NDList list = new NDList();
        for (Pointer ptr : ndPtrs) {
            list.add(new PtNDArray((PtNDManager) ndArray.getManager(), ptr));
        }
        return list;
    }

    public static NDList split(PtNDArray ndArray, long[] indices, long axis) {
        Pointer[] ndPtrs = PyTorchLibrary.LIB.torchSplit(ndArray.getHandle(), indices, axis);
        NDList list = new NDList();
        for (Pointer ptr : ndPtrs) {
            list.add(new PtNDArray((PtNDManager) ndArray.getManager(), ptr));
        }
        return list;
    }

    public static DataType getDataType(PtNDArray ndArray) {
        int dataType = PyTorchLibrary.LIB.torchDType(ndArray.getHandle());
        return DataType.values()[dataType];
    }

    public static Device getDevice(PtNDArray ndArray) {
        int[] device = PyTorchLibrary.LIB.torchDevice(ndArray.getHandle());
        return new Device(PtDeviceType.fromDeviceType(device[0]), device[1]);
    }

    public static Shape getShape(PtNDArray ndArray) {
        return new Shape(PyTorchLibrary.LIB.torchSizes(ndArray.getHandle()));
    }

    public static ByteBuffer getByteBuffer(PtNDArray ndArray) {
        ByteBuffer bb = PyTorchLibrary.LIB.torchDataPtr(ndArray.getHandle());
        bb.order(ByteOrder.nativeOrder());
        return bb;
    }

    public static void deleteNdArray(PtNDArray ndArray) {
        Pointer pointer = ndArray.getHandle();
        PyTorchLibrary.LIB.torchDeleteTensor(pointer);
    }

    public static void deleteModule(PtSymbolBlock block) {
        Pointer pointer = block.getHandle();
        PyTorchLibrary.LIB.torchDeleteModule(pointer);
    }

    public static PtSymbolBlock loadModule(PtNDManager manager, Path path) {
        Pointer handle = PyTorchLibrary.LIB.moduleLoad(path.toString());
        return new PtSymbolBlock(manager, handle);
    }

    public static void moduleEval(Pointer handle) {
        PyTorchLibrary.LIB.moduleEval(handle);
    }

    public static NDList moduleForward(Pointer handle, NDList inputs) {
        // TODO: reconsider the usage of IValue
        // Currently, only map Tensor to IValue
        Pointer[] iValueHandles =
                inputs.stream()
                        .map(
                                ele ->
                                        PyTorchLibrary.LIB.iValueCreateFromTensor(
                                                ((PtNDArray) ele).getHandle()))
                        .toArray(Pointer[]::new);
        NDArray result =
                new PtNDArray(
                        (PtNDManager) inputs.head().getManager(),
                        PyTorchLibrary.LIB.iValueToTensor(
                                PyTorchLibrary.LIB.moduleForward(handle, iValueHandles)));
        return new NDList(result);
    }
}
