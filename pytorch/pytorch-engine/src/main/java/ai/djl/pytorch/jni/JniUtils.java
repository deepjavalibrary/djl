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
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * A class containing utilities to interact with the PyTorch Engine's Java Native Interface (JNI)
 * layer.
 */
@SuppressWarnings("MissingJavadocMethod")
public final class JniUtils {

    @SuppressWarnings("PMD.UnusedPrivateField")
    private static final Logger logger = LoggerFactory.getLogger(JniUtils.class);

    private static Set<String> configs;

    private JniUtils() {}

    private static int layoutMapper(SparseFormat fmt) {
        if (fmt == SparseFormat.DENSE) {
            // Enable MKLDNN with environment variable
            return Boolean.getBoolean("ai.djl.pytorch.use_mkldnn") ? 2 : 0;
        } else if (fmt == SparseFormat.COO) {
            return 1;
        } else {
            throw new IllegalArgumentException(
                    "Current PyTorch only support SparseFormat.DENSE and SparseFormat.COO");
        }
    }

    public static void setNumInteropThreads(int threads) {
        PyTorchLibrary.LIB.torchSetNumInteropThreads(threads);
    }

    public static void setNumThreads(int threads) {
        PyTorchLibrary.LIB.torchSetNumThreads(threads);
    }

    public static Set<String> getFeatures() {
        if (configs != null) {
            return configs;
        }
        Set<String> features = new HashSet<>();
        PyTorchLibrary.LIB.torchShowConfig(features);
        configs = features;
        return configs;
    }

    public static void setSeed(long seed) {
        PyTorchLibrary.LIB.torchManualSeed(seed);
    }

    // TODO: Unchecked Datatype and device mapping
    public static PtNDArray createNdFromByteBuffer(
            PtNDManager manager,
            ByteBuffer data,
            Shape shape,
            DataType dType,
            SparseFormat fmt,
            Device device) {
        int layoutVal = layoutMapper(fmt);
        return manager.create(
                PyTorchLibrary.LIB.torchFromBlob(
                        data,
                        shape.getShape(),
                        dType.ordinal(),
                        layoutVal,
                        new int[] {
                            PtDeviceType.toDeviceType(device),
                            device.equals(Device.cpu()) ? -1 : device.getDeviceId()
                        },
                        false));
    }

    public static PtNDArray createEmptyNdArray(
            PtNDManager manager, Shape shape, DataType dType, Device device, SparseFormat fmt) {
        int layoutVal = layoutMapper(fmt);
        // TODO: set default type of require gradient
        return manager.create(
                PyTorchLibrary.LIB.torchEmpty(
                        shape.getShape(),
                        dType.ordinal(),
                        layoutVal,
                        new int[] {
                            PtDeviceType.toDeviceType(device),
                            device.equals(Device.cpu()) ? -1 : device.getDeviceId()
                        },
                        false));
    }

    public static PtNDArray createZerosNdArray(
            PtNDManager manager, Shape shape, DataType dType, Device device, SparseFormat fmt) {
        int layoutVal = layoutMapper(fmt);
        // TODO: set default type of require gradient
        return manager.create(
                PyTorchLibrary.LIB.torchZeros(
                        shape.getShape(),
                        dType.ordinal(),
                        layoutVal,
                        new int[] {
                            PtDeviceType.toDeviceType(device),
                            device.equals(Device.cpu()) ? -1 : device.getDeviceId()
                        },
                        false));
    }

    public static PtNDArray createOnesNdArray(
            PtNDManager manager, Shape shape, DataType dType, Device device, SparseFormat fmt) {
        int layoutVal = layoutMapper(fmt);
        // TODO: set default type of require gradient
        return manager.create(
                PyTorchLibrary.LIB.torchOnes(
                        shape.getShape(),
                        dType.ordinal(),
                        layoutVal,
                        new int[] {
                            PtDeviceType.toDeviceType(device),
                            device.equals(Device.cpu()) ? -1 : device.getDeviceId()
                        },
                        false));
    }

    public static PtNDArray zerosLike(
            PtNDArray array, DataType dType, Device device, SparseFormat fmt) {
        int layoutVal = layoutMapper(fmt);
        // TODO: set default type of require gradient
        return array.getManager()
                .create(
                        PyTorchLibrary.LIB.torchZerosLike(
                                array.getHandle(),
                                dType.ordinal(),
                                layoutVal,
                                new int[] {
                                    PtDeviceType.toDeviceType(device),
                                    device.equals(Device.cpu()) ? -1 : device.getDeviceId()
                                },
                                false));
    }

    public static PtNDArray onesLike(
            PtNDArray array, DataType dType, Device device, SparseFormat fmt) {
        int layoutVal = layoutMapper(fmt);
        // TODO: set default type of require gradient
        return array.getManager()
                .create(
                        PyTorchLibrary.LIB.torchOnesLike(
                                array.getHandle(),
                                dType.ordinal(),
                                layoutVal,
                                new int[] {
                                    PtDeviceType.toDeviceType(device),
                                    device.equals(Device.cpu()) ? -1 : device.getDeviceId()
                                },
                                false));
    }

    public static PtNDArray arange(
            PtNDManager manager,
            float start,
            float stop,
            float step,
            DataType dType,
            Device device,
            SparseFormat fmt) {
        int layoutVal = layoutMapper(fmt);
        return manager.create(
                PyTorchLibrary.LIB.torchArange(
                        start,
                        stop,
                        step,
                        dType.ordinal(),
                        layoutVal,
                        new int[] {
                            PtDeviceType.toDeviceType(device),
                            device.equals(Device.cpu()) ? -1 : device.getDeviceId()
                        },
                        false));
    }

    public static PtNDArray linspace(
            PtNDManager manager,
            float start,
            float stop,
            int step,
            DataType dType,
            Device device,
            SparseFormat fmt) {
        int layoutVal = layoutMapper(fmt);
        return manager.create(
                PyTorchLibrary.LIB.torchLinspace(
                        start,
                        stop,
                        step,
                        dType.ordinal(),
                        layoutVal,
                        new int[] {
                            PtDeviceType.toDeviceType(device),
                            device.equals(Device.cpu()) ? -1 : device.getDeviceId()
                        },
                        false));
    }

    public static PtNDArray to(PtNDArray ndArray, DataType dataType, Device device, boolean copy) {
        return ndArray.getManager()
                .create(
                        PyTorchLibrary.LIB.torchTo(
                                ndArray.getHandle(),
                                dataType.ordinal(),
                                new int[] {
                                    PtDeviceType.toDeviceType(device),
                                    device.equals(Device.cpu()) ? -1 : device.getDeviceId()
                                },
                                copy));
    }

    public static PtNDArray broadcast(PtNDArray ndArray, Shape shape) {
        return ndArray.getManager()
                .create(PyTorchLibrary.LIB.torchExpand(ndArray.getHandle(), shape.getShape()));
    }

    public static PtNDArray slice(PtNDArray ndArray, long dim, long start, long stop, long step) {
        return ndArray.getManager()
                .create(PyTorchLibrary.LIB.torchSlice(ndArray.getHandle(), dim, start, stop, step));
    }

    public static PtNDArray booleanMask(PtNDArray ndArray, PtNDArray indicesNd) {
        return ndArray.getManager()
                .create(
                        PyTorchLibrary.LIB.torchMaskedSelect(
                                ndArray.getHandle(), indicesNd.getHandle()));
    }

    public static PtNDArray clone(PtNDArray ndArray) {
        return ndArray.getManager().create(PyTorchLibrary.LIB.tensorClone(ndArray.getHandle()));
    }

    public static PtNDArray reshape(PtNDArray ndArray, long[] shape) {
        return ndArray.getManager()
                .create(PyTorchLibrary.LIB.torchReshape(ndArray.getHandle(), shape));
    }

    public static PtNDArray stack(NDArray[] arrays, int dim) {
        Pointer[] pointers =
                Arrays.stream(arrays)
                        .map(array -> ((PtNDArray) array).getHandle())
                        .toArray(Pointer[]::new);
        return ((PtNDManager) arrays[0].getManager())
                .create(PyTorchLibrary.LIB.torchStack(pointers, dim));
    }

    public static PtNDArray cat(NDArray[] arrays, long dim) {
        Pointer[] pointers =
                Arrays.stream(arrays)
                        .map(array -> ((PtNDArray) array).getHandle())
                        .toArray(Pointer[]::new);
        return ((PtNDManager) arrays[0].getManager())
                .create(PyTorchLibrary.LIB.torchCat(pointers, dim));
    }

    public static PtNDArray softmax(PtNDArray ndArray, long dim, DataType dTpe) {
        return ndArray.getManager()
                .create(PyTorchLibrary.LIB.torchSoftmax(ndArray.getHandle(), dim, dTpe.ordinal()));
    }

    public static PtNDArray argMax(PtNDArray ndArray) {
        return ndArray.getManager().create(PyTorchLibrary.LIB.torchArgMax(ndArray.getHandle()));
    }

    public static PtNDArray argMax(PtNDArray ndArray, long dim, boolean keepDim) {
        return ndArray.getManager()
                .create(PyTorchLibrary.LIB.torchArgMax(ndArray.getHandle(), dim, keepDim));
    }

    public static PtNDArray argMin(PtNDArray ndArray) {
        return ndArray.getManager().create(PyTorchLibrary.LIB.torchArgMin(ndArray.getHandle()));
    }

    public static PtNDArray argMin(PtNDArray ndArray, long dim, boolean keepDim) {
        return ndArray.getManager()
                .create(PyTorchLibrary.LIB.torchArgMin(ndArray.getHandle(), dim, keepDim));
    }

    public static PtNDArray argSort(PtNDArray ndArray, long dim, boolean keepDim) {
        return ndArray.getManager()
                .create(PyTorchLibrary.LIB.torchArgSort(ndArray.getHandle(), dim, keepDim));
    }

    public static PtNDArray sort(PtNDArray ndArray, long dim, boolean descending) {
        return ndArray.getManager()
                .create(PyTorchLibrary.LIB.torchSort(ndArray.getHandle(), dim, descending));
    }

    public static PtNDArray permute(PtNDArray ndArray, long[] dims) {
        return ndArray.getManager()
                .create(PyTorchLibrary.LIB.torchPermute(ndArray.getHandle(), dims));
    }

    public static PtNDArray transpose(PtNDArray ndArray, long dim1, long dim2) {
        return ndArray.getManager()
                .create(PyTorchLibrary.LIB.torchTranspose(ndArray.getHandle(), dim1, dim2));
    }

    public static boolean contentEqual(PtNDArray ndArray1, PtNDArray ndArray2) {
        return PyTorchLibrary.LIB.contentEqual(ndArray1.getHandle(), ndArray2.getHandle());
    }

    public static PtNDArray add(PtNDArray ndArray1, PtNDArray ndArray2) {
        return ndArray1.getManager()
                .create(PyTorchLibrary.LIB.torchAdd(ndArray1.getHandle(), ndArray2.getHandle()));
    }

    public static void addi(PtNDArray ndArray1, PtNDArray ndArray2) {
        PyTorchLibrary.LIB.torchAddi(ndArray1.getHandle(), ndArray2.getHandle());
    }

    public static PtNDArray sub(PtNDArray ndArray1, PtNDArray ndArray2) {
        return ndArray1.getManager()
                .create(PyTorchLibrary.LIB.torchSub(ndArray1.getHandle(), ndArray2.getHandle()));
    }

    public static void subi(PtNDArray ndArray1, PtNDArray ndArray2) {
        PyTorchLibrary.LIB.torchSubi(ndArray1.getHandle(), ndArray2.getHandle());
    }

    public static PtNDArray mul(PtNDArray ndArray1, PtNDArray ndArray2) {
        return ndArray1.getManager()
                .create(PyTorchLibrary.LIB.torchMul(ndArray1.getHandle(), ndArray2.getHandle()));
    }

    public static void muli(PtNDArray ndArray1, PtNDArray ndArray2) {
        PyTorchLibrary.LIB.torchMuli(ndArray1.getHandle(), ndArray2.getHandle());
    }

    public static PtNDArray div(PtNDArray ndArray1, PtNDArray ndArray2) {
        return ndArray1.getManager()
                .create(PyTorchLibrary.LIB.torchDiv(ndArray1.getHandle(), ndArray2.getHandle()));
    }

    public static void divi(PtNDArray ndArray1, PtNDArray ndArray2) {
        PyTorchLibrary.LIB.torchDivi(ndArray1.getHandle(), ndArray2.getHandle());
    }

    public static PtNDArray remainder(PtNDArray ndArray1, PtNDArray ndArray2) {
        return ndArray1.getManager()
                .create(
                        PyTorchLibrary.LIB.torchRemainder(
                                ndArray1.getHandle(), ndArray2.getHandle()));
    }

    public static void remainderi(PtNDArray ndArray1, PtNDArray ndArray2) {
        PyTorchLibrary.LIB.torchRemainderi(ndArray1.getHandle(), ndArray2.getHandle());
    }

    public static PtNDArray pow(PtNDArray ndArray1, PtNDArray ndArray2) {
        return ndArray1.getManager()
                .create(PyTorchLibrary.LIB.torchPow(ndArray1.getHandle(), ndArray2.getHandle()));
    }

    public static void powi(PtNDArray ndArray1, PtNDArray ndArray2) {
        PyTorchLibrary.LIB.torchPowi(ndArray1.getHandle(), ndArray2.getHandle());
    }

    public static PtNDArray logicalXor(PtNDArray ndArray1, PtNDArray ndArray2) {
        return ndArray1.getManager()
                .create(
                        PyTorchLibrary.LIB.torchLogicalXor(
                                ndArray1.getHandle(), ndArray2.getHandle()));
    }

    public static PtNDArray logicalNot(PtNDArray ndArray) {
        return ndArray.getManager().create(PyTorchLibrary.LIB.torchLogicalNot(ndArray.getHandle()));
    }

    public static PtNDArray matmul(PtNDArray ndArray1, PtNDArray ndArray2) {
        return ndArray1.getManager()
                .create(PyTorchLibrary.LIB.torchMatmul(ndArray1.getHandle(), ndArray2.getHandle()));
    }

    public static PtNDArray max(PtNDArray ndArray1, PtNDArray ndArray2) {
        return ndArray1.getManager()
                .create(PyTorchLibrary.LIB.torchMax(ndArray1.getHandle(), ndArray2.getHandle()));
    }

    public static PtNDArray max(PtNDArray ndArray) {
        return ndArray.getManager().create(PyTorchLibrary.LIB.torchMax(ndArray.getHandle()));
    }

    public static PtNDArray max(PtNDArray ndArray, long dim, boolean keepDim) {
        return ndArray.getManager()
                .create(PyTorchLibrary.LIB.torchMax(ndArray.getHandle(), dim, keepDim));
    }

    public static PtNDArray min(PtNDArray ndArray1, PtNDArray ndArray2) {
        return ndArray1.getManager()
                .create(PyTorchLibrary.LIB.torchMin(ndArray1.getHandle(), ndArray2.getHandle()));
    }

    public static PtNDArray min(PtNDArray ndArray) {
        return ndArray.getManager().create(PyTorchLibrary.LIB.torchMin(ndArray.getHandle()));
    }

    public static PtNDArray min(PtNDArray ndArray, long dim, boolean keepDim) {
        return ndArray.getManager()
                .create(PyTorchLibrary.LIB.torchMin(ndArray.getHandle(), dim, keepDim));
    }

    public static PtNDArray mean(PtNDArray ndArray) {
        return ndArray.getManager().create(PyTorchLibrary.LIB.torchMean(ndArray.getHandle()));
    }

    public static PtNDArray mean(PtNDArray ndArray, long dim, boolean keepDim) {
        return ndArray.getManager()
                .create(PyTorchLibrary.LIB.torchMean(ndArray.getHandle(), dim, keepDim));
    }

    public static PtNDArray sum(PtNDArray ndArray) {
        return ndArray.getManager().create(PyTorchLibrary.LIB.torchSum(ndArray.getHandle()));
    }

    public static PtNDArray sum(PtNDArray ndArray, long[] dims, boolean keepDim) {
        return ndArray.getManager()
                .create(PyTorchLibrary.LIB.torchSum(ndArray.getHandle(), dims, keepDim));
    }

    public static NDList split(PtNDArray ndArray, long size, long axis) {
        Pointer[] ndPtrs = PyTorchLibrary.LIB.torchSplit(ndArray.getHandle(), size, axis);
        NDList list = new NDList();
        for (Pointer ptr : ndPtrs) {
            list.add(ndArray.getManager().create(ptr));
        }
        return list;
    }

    public static NDList split(PtNDArray ndArray, long[] indices, long axis) {
        Pointer[] ndPtrs = PyTorchLibrary.LIB.torchSplit(ndArray.getHandle(), indices, axis);
        NDList list = new NDList();
        for (Pointer ptr : ndPtrs) {
            list.add(ndArray.getManager().create(ptr));
        }
        return list;
    }

    public static PtNDArray squeeze(PtNDArray ndArray) {
        return ndArray.getManager().create(PyTorchLibrary.LIB.torchSqueeze(ndArray.getHandle()));
    }

    public static PtNDArray squeeze(PtNDArray ndArray, long dim) {
        return ndArray.getManager()
                .create(PyTorchLibrary.LIB.torchSqueeze(ndArray.getHandle(), dim));
    }

    public static PtNDArray unsqueeze(PtNDArray ndArray, long dim) {
        return ndArray.getManager()
                .create(PyTorchLibrary.LIB.torchUnsqueeze(ndArray.getHandle(), dim));
    }

    public static PtNDArray flatten(PtNDArray ndArray, long startDim, long endDim) {
        return ndArray.getManager()
                .create(PyTorchLibrary.LIB.torchFlatten(ndArray.getHandle(), startDim, endDim));
    }

    public static PtNDArray abs(PtNDArray ndArray) {
        return ndArray.getManager().create(PyTorchLibrary.LIB.torchAbs(ndArray.getHandle()));
    }

    public static PtNDArray floor(PtNDArray ndArray) {
        return ndArray.getManager().create(PyTorchLibrary.LIB.torchFloor(ndArray.getHandle()));
    }

    public static PtNDArray ceil(PtNDArray ndArray) {
        return ndArray.getManager().create(PyTorchLibrary.LIB.torchCeil(ndArray.getHandle()));
    }

    public static PtNDArray round(PtNDArray ndArray) {
        return ndArray.getManager().create(PyTorchLibrary.LIB.torchRound(ndArray.getHandle()));
    }

    public static PtNDArray trunc(PtNDArray ndArray) {
        return ndArray.getManager().create(PyTorchLibrary.LIB.torchTrunc(ndArray.getHandle()));
    }

    public static PtNDArray clip(PtNDArray ndArray, Number min, Number max) {
        PtNDArray minNd = (PtNDArray) ndArray.getManager().create(min);
        PtNDArray maxNd = (PtNDArray) ndArray.getManager().create(max);
        return ndArray.getManager()
                .create(
                        PyTorchLibrary.LIB.torchClamp(
                                ndArray.getHandle(), minNd.getHandle(), maxNd.getHandle()));
    }

    public static PtNDArray exp(PtNDArray ndArray) {
        return ndArray.getManager().create(PyTorchLibrary.LIB.torchExp(ndArray.getHandle()));
    }

    public static PtNDArray log(PtNDArray ndArray) {
        return ndArray.getManager().create(PyTorchLibrary.LIB.torchLog(ndArray.getHandle()));
    }

    public static PtNDArray log10(PtNDArray ndArray) {
        return ndArray.getManager().create(PyTorchLibrary.LIB.torchLog10(ndArray.getHandle()));
    }

    public static PtNDArray log2(PtNDArray ndArray) {
        return ndArray.getManager().create(PyTorchLibrary.LIB.torchLog2(ndArray.getHandle()));
    }

    public static PtNDArray sin(PtNDArray ndArray) {
        return ndArray.getManager().create(PyTorchLibrary.LIB.torchSin(ndArray.getHandle()));
    }

    public static PtNDArray cos(PtNDArray ndArray) {
        return ndArray.getManager().create(PyTorchLibrary.LIB.torchCos(ndArray.getHandle()));
    }

    public static PtNDArray tan(PtNDArray ndArray) {
        return ndArray.getManager().create(PyTorchLibrary.LIB.torchTan(ndArray.getHandle()));
    }

    public static PtNDArray asin(PtNDArray ndArray) {
        return ndArray.getManager().create(PyTorchLibrary.LIB.torchASin(ndArray.getHandle()));
    }

    public static PtNDArray acos(PtNDArray ndArray) {
        return ndArray.getManager().create(PyTorchLibrary.LIB.torchAcos(ndArray.getHandle()));
    }

    public static PtNDArray atan(PtNDArray ndArray) {
        return ndArray.getManager().create(PyTorchLibrary.LIB.torchAtan(ndArray.getHandle()));
    }

    public static PtNDArray sqrt(PtNDArray ndArray) {
        return ndArray.getManager().create(PyTorchLibrary.LIB.torchSqrt(ndArray.getHandle()));
    }

    public static PtNDArray sinh(PtNDArray ndArray) {
        return ndArray.getManager().create(PyTorchLibrary.LIB.torchSinh(ndArray.getHandle()));
    }

    public static PtNDArray cosh(PtNDArray ndArray) {
        return ndArray.getManager().create(PyTorchLibrary.LIB.torchCosh(ndArray.getHandle()));
    }

    public static PtNDArray tanh(PtNDArray ndArray) {
        return ndArray.getManager().create(PyTorchLibrary.LIB.torchTanh(ndArray.getHandle()));
    }

    public static PtNDArray all(PtNDArray ndArray) {
        return ndArray.getManager().create(PyTorchLibrary.LIB.torchAll(ndArray.getHandle()));
    }

    public static PtNDArray any(PtNDArray ndArray) {
        return ndArray.getManager().create(PyTorchLibrary.LIB.torchAny(ndArray.getHandle()));
    }

    public static PtNDArray none(PtNDArray ndArray) {
        return ndArray.getManager().create(PyTorchLibrary.LIB.torchNone(ndArray.getHandle()));
    }

    public static PtNDArray eq(PtNDArray self, PtNDArray other) {
        return self.getManager()
                .create(PyTorchLibrary.LIB.torchEq(self.getHandle(), other.getHandle()));
    }

    public static PtNDArray neq(PtNDArray self, PtNDArray other) {
        return self.getManager()
                .create(PyTorchLibrary.LIB.torchNeq(self.getHandle(), other.getHandle()));
    }

    public static PtNDArray gt(PtNDArray self, PtNDArray other) {
        return self.getManager()
                .create(PyTorchLibrary.LIB.torchGt(self.getHandle(), other.getHandle()));
    }

    public static PtNDArray gte(PtNDArray self, PtNDArray other) {
        return self.getManager()
                .create(PyTorchLibrary.LIB.torchGte(self.getHandle(), other.getHandle()));
    }

    public static PtNDArray lt(PtNDArray self, PtNDArray other) {
        return self.getManager()
                .create(PyTorchLibrary.LIB.torchLt(self.getHandle(), other.getHandle()));
    }

    public static PtNDArray lte(PtNDArray self, PtNDArray other) {
        return self.getManager()
                .create(PyTorchLibrary.LIB.torchLte(self.getHandle(), other.getHandle()));
    }

    public static PtNDArray neg(PtNDArray ndArray) {
        return ndArray.getManager().create(PyTorchLibrary.LIB.torchNeg(ndArray.getHandle()));
    }

    public static void negi(PtNDArray ndArray) {
        PyTorchLibrary.LIB.torchNegi(ndArray.getHandle());
    }

    public static PtNDArray normal(
            PtNDManager manager,
            double mean,
            double std,
            Shape size,
            DataType dataType,
            Device device) {
        return manager.create(
                PyTorchLibrary.LIB.atNormal(
                        mean,
                        std,
                        size.getShape(),
                        dataType.ordinal(),
                        layoutMapper(SparseFormat.DENSE),
                        new int[] {
                            PtDeviceType.toDeviceType(device),
                            device.equals(Device.cpu()) ? -1 : device.getDeviceId()
                        },
                        false));
    }

    public static PtNDArray uniform(
            PtNDManager manager,
            double low,
            double high,
            Shape size,
            DataType dataType,
            Device device) {
        return manager.create(
                PyTorchLibrary.LIB.tensorUniform(
                        low,
                        high,
                        size.getShape(),
                        dataType.ordinal(),
                        layoutMapper(SparseFormat.DENSE),
                        new int[] {
                            PtDeviceType.toDeviceType(device),
                            device.equals(Device.cpu()) ? -1 : device.getDeviceId()
                        },
                        false));
    }

    public static PtNDArray eye(
            PtNDManager manager, int n, int m, DataType dataType, Device device, SparseFormat fmt) {
        return manager.create(
                PyTorchLibrary.LIB.torchEye(
                        n,
                        m,
                        dataType.ordinal(),
                        layoutMapper(fmt),
                        new int[] {
                            PtDeviceType.toDeviceType(device),
                            device.equals(Device.cpu()) ? -1 : device.getDeviceId()
                        },
                        false));
    }

    public static PtNDArray upsampleBilinear2d(
            PtNDArray ndArray, long[] size, boolean alignCorners) {
        return ndArray.getManager()
                .create(
                        PyTorchLibrary.LIB.torchUpsampleBilinear2d(
                                ndArray.getHandle(), size, alignCorners));
    }

    public static DataType getDataType(PtNDArray ndArray) {
        int dataType = PyTorchLibrary.LIB.torchDType(ndArray.getHandle());
        return DataType.values()[dataType];
    }

    public static Device getDevice(PtNDArray ndArray) {
        int[] device = PyTorchLibrary.LIB.torchDevice(ndArray.getHandle());
        String deviceType = PtDeviceType.fromDeviceType(device[0]);
        return Device.of(deviceType, device[1]);
    }

    public static SparseFormat getSparseFormat(PtNDArray ndArray) {
        int layout = PyTorchLibrary.LIB.torchLayout(ndArray.getHandle());
        if (layout == 0) {
            return SparseFormat.DENSE;
        } else if (layout == 1) {
            return SparseFormat.COO;
        } else {
            throw new UnsupportedOperationException("Unsupported data format");
        }
    }

    public static Shape getShape(PtNDArray ndArray) {
        return new Shape(PyTorchLibrary.LIB.torchSizes(ndArray.getHandle()));
    }

    public static ByteBuffer getByteBuffer(PtNDArray ndArray) {
        // Operation is CPU only
        if (!ndArray.getDevice().equals(Device.cpu())) {
            ndArray = ndArray.toDevice(Device.cpu(), false);
        }
        return ByteBuffer.wrap(PyTorchLibrary.LIB.torchDataPtr(ndArray.getHandle()))
                .order(ByteOrder.nativeOrder());
    }

    public static void deleteNdArray(Pointer handle) {
        PyTorchLibrary.LIB.torchDeleteTensor(handle);
    }

    public static void deleteModule(PtSymbolBlock block) {
        PyTorchLibrary.LIB.torchDeleteModule(block.getHandle());
    }

    public static PtSymbolBlock loadModule(PtNDManager manager, Path path, Device device) {
        Pointer handle =
                PyTorchLibrary.LIB.moduleLoad(
                        path.toString(),
                        new int[] {
                            PtDeviceType.toDeviceType(device),
                            device.equals(Device.cpu()) ? -1 : device.getDeviceId()
                        });
        return new PtSymbolBlock(manager, handle);
    }

    public static void enableInferenceMode(PtSymbolBlock block) {
        PyTorchLibrary.LIB.moduleEval(block.getHandle());
    }
}
