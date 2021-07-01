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
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.ndarray.types.SparseFormat;
import ai.djl.nn.recurrent.RNN;
import ai.djl.pytorch.engine.PtDeviceType;
import ai.djl.pytorch.engine.PtNDArray;
import ai.djl.pytorch.engine.PtNDManager;
import ai.djl.pytorch.engine.PtSymbolBlock;
import java.io.DataInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
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

    private static final Logger logger = LoggerFactory.getLogger(JniUtils.class);

    private static Set<String> configs;

    private static final int NULL_PTR = 0;

    private static final int BYTE_LENGTH = 4194304;

    private JniUtils() {}

    private static int layoutMapper(SparseFormat fmt, Device device) {
        if (fmt == SparseFormat.DENSE) {
            // Enable MKLDNN with environment variable
            // Using MKLDNN with GPU would throw exception on libtorch
            if (Boolean.getBoolean("ai.djl.pytorch.use_mkldnn") && !device.equals(Device.gpu())) {
                return 2;
            }
            return 0;
        } else if (fmt == SparseFormat.COO) {
            return 1;
        } else {
            throw new IllegalArgumentException(
                    "Current PyTorch only support SparseFormat.DENSE and SparseFormat.COO");
        }
    }

    public static int getNumInteropThreads() {
        return PyTorchLibrary.LIB.torchGetNumInteropThreads();
    }

    public static int getNumThreads() {
        return PyTorchLibrary.LIB.torchGetNumThreads();
    }

    public static void setNumInteropThreads(int threads) {
        PyTorchLibrary.LIB.torchSetNumInteropThreads(threads);
    }

    public static void setNumThreads(int threads) {
        PyTorchLibrary.LIB.torchSetNumThreads(threads);
    }

    public static synchronized Set<String> getFeatures() {
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

    /**
     * Calls this method to start profile the area you are interested in.
     *
     * <p>Example usage
     *
     * <pre>
     *      JniUtils.startProfile(false, true, true);
     *      Predictor.predict(img);
     *      JniUtils.stopProfile(outputFile)
     * </pre>
     *
     * @param useCuda Enables timing of CUDA events as well using the cudaEvent API.
     * @param recordShape If shapes recording is set, information about input dimensions will be
     *     collected
     * @param profileMemory Whether to report memory usage
     */
    public static synchronized void startProfile(
            boolean useCuda, boolean recordShape, boolean profileMemory) {
        PyTorchLibrary.LIB.torchStartProfile(useCuda, recordShape, profileMemory);
    }

    public static synchronized void stopProfile(String outputFile) {
        PyTorchLibrary.LIB.torchStopProfile(outputFile);
    }

    // TODO: Unchecked Datatype and device mapping
    public static PtNDArray createNdFromByteBuffer(
            PtNDManager manager,
            ByteBuffer data,
            Shape shape,
            DataType dType,
            SparseFormat fmt,
            Device device) {
        int layout = layoutMapper(fmt, device);
        long handle =
                PyTorchLibrary.LIB.torchFromBlob(
                        data,
                        shape.getShape(),
                        dType.ordinal(),
                        layout,
                        new int[] {PtDeviceType.toDeviceType(device), device.getDeviceId()},
                        false);

        if (layout == 1 || layout == 2 || Device.Type.GPU.equals(device.getDeviceType())) {
            // MKLDNN & COO & GPU device will explicitly make a copy in native code
            // so we don't want to hold a reference on Java side
            return new PtNDArray(manager, handle);
        }
        return new PtNDArray(manager, handle, data);
    }

    public static PtNDArray createEmptyNdArray(
            PtNDManager manager, Shape shape, DataType dType, Device device, SparseFormat fmt) {
        int layoutVal = layoutMapper(fmt, device);
        return new PtNDArray(
                manager,
                PyTorchLibrary.LIB.torchEmpty(
                        shape.getShape(),
                        dType.ordinal(),
                        layoutVal,
                        new int[] {PtDeviceType.toDeviceType(device), device.getDeviceId()},
                        false));
    }

    public static PtNDArray createZerosNdArray(
            PtNDManager manager, Shape shape, DataType dType, Device device, SparseFormat fmt) {
        int layoutVal = layoutMapper(fmt, device);
        return new PtNDArray(
                manager,
                PyTorchLibrary.LIB.torchZeros(
                        shape.getShape(),
                        dType.ordinal(),
                        layoutVal,
                        new int[] {PtDeviceType.toDeviceType(device), device.getDeviceId()},
                        false));
    }

    public static PtNDArray createOnesNdArray(
            PtNDManager manager, Shape shape, DataType dType, Device device, SparseFormat fmt) {
        int layoutVal = layoutMapper(fmt, device);
        return new PtNDArray(
                manager,
                PyTorchLibrary.LIB.torchOnes(
                        shape.getShape(),
                        dType.ordinal(),
                        layoutVal,
                        new int[] {PtDeviceType.toDeviceType(device), device.getDeviceId()},
                        false));
    }

    public static PtNDArray full(
            PtNDManager manager,
            Shape shape,
            double fillValue,
            DataType dType,
            Device device,
            SparseFormat fmt) {
        int layoutVal = layoutMapper(fmt, device);
        return new PtNDArray(
                manager,
                PyTorchLibrary.LIB.torchFull(
                        shape.getShape(),
                        fillValue,
                        dType.ordinal(),
                        layoutVal,
                        new int[] {PtDeviceType.toDeviceType(device), device.getDeviceId()},
                        false));
    }

    public static PtNDArray zerosLike(
            PtNDArray array, DataType dType, Device device, SparseFormat fmt) {
        int layoutVal = layoutMapper(fmt, device);
        return new PtNDArray(
                array.getManager(),
                PyTorchLibrary.LIB.torchZerosLike(
                        array.getHandle(),
                        dType.ordinal(),
                        layoutVal,
                        new int[] {PtDeviceType.toDeviceType(device), device.getDeviceId()},
                        false));
    }

    public static PtNDArray onesLike(
            PtNDArray array, DataType dType, Device device, SparseFormat fmt) {
        int layoutVal = layoutMapper(fmt, device);
        return new PtNDArray(
                array.getManager(),
                PyTorchLibrary.LIB.torchOnesLike(
                        array.getHandle(),
                        dType.ordinal(),
                        layoutVal,
                        new int[] {PtDeviceType.toDeviceType(device), device.getDeviceId()},
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
        int layoutVal = layoutMapper(fmt, device);
        return new PtNDArray(
                manager,
                PyTorchLibrary.LIB.torchArange(
                        start,
                        stop,
                        step,
                        dType.ordinal(),
                        layoutVal,
                        new int[] {PtDeviceType.toDeviceType(device), device.getDeviceId()},
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
        int layoutVal = layoutMapper(fmt, device);
        return new PtNDArray(
                manager,
                PyTorchLibrary.LIB.torchLinspace(
                        start,
                        stop,
                        step,
                        dType.ordinal(),
                        layoutVal,
                        new int[] {PtDeviceType.toDeviceType(device), device.getDeviceId()},
                        false));
    }

    public static PtNDArray createSparseCoo(PtNDArray indices, PtNDArray values, Shape shape) {
        return new PtNDArray(
                values.getManager(),
                PyTorchLibrary.LIB.torchSparseCoo(
                        shape.getShape(), indices.getHandle(), values.getHandle(), false));
    }

    public static PtNDArray to(PtNDArray ndArray, DataType dataType, Device device) {
        PtNDManager manager = ndArray.getManager();
        // the device of the manager should always match the one in NDArray which the manager attach
        // to
        if (!device.equals(manager.getDevice())) {
            manager = manager.newSubManager(device);
        }
        return new PtNDArray(
                manager,
                PyTorchLibrary.LIB.torchTo(
                        ndArray.getHandle(),
                        dataType.ordinal(),
                        new int[] {PtDeviceType.toDeviceType(device), device.getDeviceId()}));
    }

    public static PtNDArray toSparse(PtNDArray ndArray) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchToSparse(ndArray.getHandle()));
    }

    public static PtNDArray toDense(PtNDArray ndArray) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchToDense(ndArray.getHandle()));
    }

    public static PtNDArray broadcast(PtNDArray ndArray, Shape shape) {
        return new PtNDArray(
                ndArray.getManager(),
                PyTorchLibrary.LIB.torchExpand(ndArray.getHandle(), shape.getShape()));
    }

    public static PtNDArray slice(PtNDArray ndArray, long dim, long start, long stop, long step) {
        return new PtNDArray(
                ndArray.getManager(),
                PyTorchLibrary.LIB.torchSlice(ndArray.getHandle(), dim, start, stop, step));
    }

    public static PtNDArray index(
            PtNDArray ndArray, long[] minIndices, long[] maxIndices, long[] stepIndices) {
        return new PtNDArray(
                ndArray.getManager(),
                PyTorchLibrary.LIB.torchIndex(
                        ndArray.getHandle(), minIndices, maxIndices, stepIndices));
    }

    public static void indexSet(
            PtNDArray ndArray,
            PtNDArray value,
            long[] minIndices,
            long[] maxIndices,
            long[] stepIndices) {
        PyTorchLibrary.LIB.torchIndexPut(
                ndArray.getHandle(), value.getHandle(), minIndices, maxIndices, stepIndices);
    }

    public static void set(PtNDArray self, ByteBuffer data) {
        // Note the ByteBuffer here is directByteBuffer
        PyTorchLibrary.LIB.torchSet(self.getHandle(), data);
    }

    public static PtNDArray pick(PtNDArray ndArray, PtNDArray index, long dim) {
        Shape indexShape = index.getShape();
        Shape ndShape = ndArray.getShape();
        int shapeDims = indexShape.dimension();
        int ndDims = ndShape.dimension();
        if (shapeDims != ndDims) {
            for (int i = 0; i < ndDims - shapeDims; ++i) {
                if (indexShape.equals(ndShape.slice(i, shapeDims))) {
                    long[] shapes = indexShape.getShape();
                    long[] newShape = new long[ndDims];
                    Arrays.fill(newShape, 0, i, 1L);
                    Arrays.fill(newShape, i, i + shapes.length, shapes[i]);
                    Arrays.fill(newShape, i + shapes.length, ndDims, 1L);
                    indexShape = new Shape(newShape);
                    break;
                }
            }
            if (indexShape.equals(index.getShape())) {
                throw new IllegalArgumentException(
                        "expand shape failed! Cannot expand from " + indexShape + "to " + ndShape);
            }
            index = index.reshape(indexShape);
        }
        if (index.getDataType() != DataType.INT64) {
            index = index.toType(DataType.INT64, true);
        }
        return new PtNDArray(
                ndArray.getManager(),
                PyTorchLibrary.LIB.torchGather(ndArray.getHandle(), index.getHandle(), dim, false));
    }

    public static PtNDArray where(PtNDArray condition, PtNDArray self, PtNDArray other) {
        return new PtNDArray(
                self.getManager(),
                PyTorchLibrary.LIB.torchWhere(
                        condition.getHandle(), self.getHandle(), other.getHandle()));
    }

    public static PtNDArray booleanMask(PtNDArray ndArray, PtNDArray indicesNd) {
        return new PtNDArray(
                ndArray.getManager(),
                PyTorchLibrary.LIB.torchMaskedSelect(ndArray.getHandle(), indicesNd.getHandle()));
    }

    public static void booleanMaskSet(PtNDArray ndArray, PtNDArray value, PtNDArray indicesNd) {
        PyTorchLibrary.LIB.torchMaskedPut(
                ndArray.getHandle(), value.getHandle(), indicesNd.getHandle());
    }

    public static PtNDArray getItem(PtNDArray ndArray, long[] indices) {
        // use a specialized API here
        // due to significant performance gain
        // for commonly used data loading call
        if (indices.length == 1) {
            return new PtNDArray(
                    ndArray.getManager(),
                    PyTorchLibrary.LIB.torchGetItem(ndArray.getHandle(), indices[0]));
        }
        return new PtNDArray(
                ndArray.getManager(),
                PyTorchLibrary.LIB.torchGetItem(ndArray.getHandle(), indices));
    }

    public static PtNDArray clone(PtNDArray ndArray) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.tensorClone(ndArray.getHandle()));
    }

    public static PtNDArray reshape(PtNDArray ndArray, long[] shape) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchReshape(ndArray.getHandle(), shape));
    }

    public static PtNDArray stack(PtNDArray[] arrays, int dim) {
        long[] pointers = Arrays.stream(arrays).mapToLong(PtNDArray::getHandle).toArray();
        return new PtNDArray(arrays[0].getManager(), PyTorchLibrary.LIB.torchStack(pointers, dim));
    }

    public static PtNDArray cat(PtNDArray[] arrays, long dim) {
        long[] pointers = Arrays.stream(arrays).mapToLong(PtNDArray::getHandle).toArray();
        return new PtNDArray(arrays[0].getManager(), PyTorchLibrary.LIB.torchCat(pointers, dim));
    }

    public static PtNDArray tile(PtNDArray ndArray, long[] repeats) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchRepeat(ndArray.getHandle(), repeats));
    }

    public static PtNDArray repeat(PtNDArray ndArray, long repeat, long dim) {
        return new PtNDArray(
                ndArray.getManager(),
                PyTorchLibrary.LIB.torchRepeatInterleave(ndArray.getHandle(), repeat, dim));
    }

    public static PtNDArray softmax(PtNDArray ndArray, long dim, DataType dTpe) {
        return new PtNDArray(
                ndArray.getManager(),
                PyTorchLibrary.LIB.torchSoftmax(ndArray.getHandle(), dim, dTpe.ordinal()));
    }

    public static PtNDArray logSoftmax(PtNDArray ndArray, long dim, DataType dTpe) {
        return new PtNDArray(
                ndArray.getManager(),
                PyTorchLibrary.LIB.torchLogSoftmax(ndArray.getHandle(), dim, dTpe.ordinal()));
    }

    public static PtNDArray argMax(PtNDArray ndArray) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchArgMax(ndArray.getHandle()));
    }

    public static PtNDArray argMax(PtNDArray ndArray, long dim, boolean keepDim) {
        return new PtNDArray(
                ndArray.getManager(),
                PyTorchLibrary.LIB.torchArgMax(ndArray.getHandle(), dim, keepDim));
    }

    public static PtNDArray argMin(PtNDArray ndArray) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchArgMin(ndArray.getHandle()));
    }

    public static PtNDArray argMin(PtNDArray ndArray, long dim, boolean keepDim) {
        return new PtNDArray(
                ndArray.getManager(),
                PyTorchLibrary.LIB.torchArgMin(ndArray.getHandle(), dim, keepDim));
    }

    public static PtNDArray argSort(PtNDArray ndArray, long dim, boolean keepDim) {
        return new PtNDArray(
                ndArray.getManager(),
                PyTorchLibrary.LIB.torchArgSort(ndArray.getHandle(), dim, keepDim));
    }

    public static PtNDArray sort(PtNDArray ndArray, long dim, boolean descending) {
        return new PtNDArray(
                ndArray.getManager(),
                PyTorchLibrary.LIB.torchSort(ndArray.getHandle(), dim, descending));
    }

    public static PtNDArray permute(PtNDArray ndArray, long[] dims) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchPermute(ndArray.getHandle(), dims));
    }

    public static PtNDArray flip(PtNDArray ndArray, long[] dims) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchFlip(ndArray.getHandle(), dims));
    }

    public static PtNDArray transpose(PtNDArray ndArray, long dim1, long dim2) {
        return new PtNDArray(
                ndArray.getManager(),
                PyTorchLibrary.LIB.torchTranspose(ndArray.getHandle(), dim1, dim2));
    }

    public static boolean contentEqual(PtNDArray ndArray1, PtNDArray ndArray2) {
        return PyTorchLibrary.LIB.contentEqual(ndArray1.getHandle(), ndArray2.getHandle());
    }

    public static PtNDArray add(PtNDArray ndArray1, PtNDArray ndArray2) {
        return new PtNDArray(
                ndArray1.getManager(),
                PyTorchLibrary.LIB.torchAdd(ndArray1.getHandle(), ndArray2.getHandle()));
    }

    public static void addi(PtNDArray ndArray1, PtNDArray ndArray2) {
        PyTorchLibrary.LIB.torchAddi(ndArray1.getHandle(), ndArray2.getHandle());
    }

    public static PtNDArray sub(PtNDArray ndArray1, PtNDArray ndArray2) {
        return new PtNDArray(
                ndArray1.getManager(),
                PyTorchLibrary.LIB.torchSub(ndArray1.getHandle(), ndArray2.getHandle()));
    }

    public static void subi(PtNDArray ndArray1, PtNDArray ndArray2) {
        PyTorchLibrary.LIB.torchSubi(ndArray1.getHandle(), ndArray2.getHandle());
    }

    public static PtNDArray mul(PtNDArray ndArray1, PtNDArray ndArray2) {
        return new PtNDArray(
                ndArray1.getManager(),
                PyTorchLibrary.LIB.torchMul(ndArray1.getHandle(), ndArray2.getHandle()));
    }

    public static void muli(PtNDArray ndArray1, PtNDArray ndArray2) {
        PyTorchLibrary.LIB.torchMuli(ndArray1.getHandle(), ndArray2.getHandle());
    }

    public static PtNDArray div(PtNDArray ndArray1, PtNDArray ndArray2) {
        return new PtNDArray(
                ndArray1.getManager(),
                PyTorchLibrary.LIB.torchTrueDivide(ndArray1.getHandle(), ndArray2.getHandle()));
    }

    public static void divi(PtNDArray ndArray1, PtNDArray ndArray2) {
        PyTorchLibrary.LIB.torchTrueDividei(ndArray1.getHandle(), ndArray2.getHandle());
    }

    public static PtNDArray remainder(PtNDArray ndArray1, PtNDArray ndArray2) {
        return new PtNDArray(
                ndArray1.getManager(),
                PyTorchLibrary.LIB.torchRemainder(ndArray1.getHandle(), ndArray2.getHandle()));
    }

    public static void remainderi(PtNDArray ndArray1, PtNDArray ndArray2) {
        PyTorchLibrary.LIB.torchRemainderi(ndArray1.getHandle(), ndArray2.getHandle());
    }

    public static PtNDArray pow(PtNDArray ndArray1, PtNDArray ndArray2) {
        return new PtNDArray(
                ndArray1.getManager(),
                PyTorchLibrary.LIB.torchPow(ndArray1.getHandle(), ndArray2.getHandle()));
    }

    public static void powi(PtNDArray ndArray1, PtNDArray ndArray2) {
        PyTorchLibrary.LIB.torchPowi(ndArray1.getHandle(), ndArray2.getHandle());
    }

    public static PtNDArray sign(PtNDArray ndArray) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchSign(ndArray.getHandle()));
    }

    public static void signi(PtNDArray ndArray) {
        PyTorchLibrary.LIB.torchSigni(ndArray.getHandle());
    }

    public static PtNDArray logicalAnd(PtNDArray ndArray1, PtNDArray ndArray2) {
        return new PtNDArray(
                ndArray1.getManager(),
                PyTorchLibrary.LIB.torchLogicalAnd(ndArray1.getHandle(), ndArray2.getHandle()));
    }

    public static PtNDArray logicalOr(PtNDArray ndArray1, PtNDArray ndArray2) {
        return new PtNDArray(
                ndArray1.getManager(),
                PyTorchLibrary.LIB.torchLogicalOr(ndArray1.getHandle(), ndArray2.getHandle()));
    }

    public static PtNDArray logicalXor(PtNDArray ndArray1, PtNDArray ndArray2) {
        return new PtNDArray(
                ndArray1.getManager(),
                PyTorchLibrary.LIB.torchLogicalXor(ndArray1.getHandle(), ndArray2.getHandle()));
    }

    public static PtNDArray logicalNot(PtNDArray ndArray) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchLogicalNot(ndArray.getHandle()));
    }

    public static PtNDArray matmul(PtNDArray ndArray1, PtNDArray ndArray2) {
        return new PtNDArray(
                ndArray1.getManager(),
                PyTorchLibrary.LIB.torchMatmul(ndArray1.getHandle(), ndArray2.getHandle()));
    }

    public static PtNDArray dot(PtNDArray ndArray1, PtNDArray ndArray2) {
        if (ndArray1.getShape().dimension() == 1) {
            return new PtNDArray(
                    ndArray1.getManager(),
                    PyTorchLibrary.LIB.torchDot(ndArray1.getHandle(), ndArray2.getHandle()));
        }
        return new PtNDArray(
                ndArray1.getManager(),
                PyTorchLibrary.LIB.torchMatmul(ndArray1.getHandle(), ndArray2.getHandle()));
    }

    public static PtNDArray max(PtNDArray ndArray1, PtNDArray ndArray2) {
        return new PtNDArray(
                ndArray1.getManager(),
                PyTorchLibrary.LIB.torchMaximum(ndArray1.getHandle(), ndArray2.getHandle()));
    }

    public static PtNDArray max(PtNDArray ndArray) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchMax(ndArray.getHandle()));
    }

    public static PtNDArray max(PtNDArray ndArray, long dim, boolean keepDim) {
        return new PtNDArray(
                ndArray.getManager(),
                PyTorchLibrary.LIB.torchMax(ndArray.getHandle(), dim, keepDim));
    }

    public static PtNDArray min(PtNDArray ndArray1, PtNDArray ndArray2) {
        return new PtNDArray(
                ndArray1.getManager(),
                PyTorchLibrary.LIB.torchMinimum(ndArray1.getHandle(), ndArray2.getHandle()));
    }

    public static PtNDArray min(PtNDArray ndArray) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchMin(ndArray.getHandle()));
    }

    public static PtNDArray min(PtNDArray ndArray, long dim, boolean keepDim) {
        return new PtNDArray(
                ndArray.getManager(),
                PyTorchLibrary.LIB.torchMin(ndArray.getHandle(), dim, keepDim));
    }

    public static PtNDArray mean(PtNDArray ndArray) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchMean(ndArray.getHandle()));
    }

    public static PtNDArray mean(PtNDArray ndArray, long dim, boolean keepDim) {
        return new PtNDArray(
                ndArray.getManager(),
                PyTorchLibrary.LIB.torchMean(ndArray.getHandle(), dim, keepDim));
    }

    public static PtNDArray rot90(PtNDArray ndArray, int times, int[] axes) {
        long[] longaxes = Arrays.stream(axes).mapToLong(i -> i).toArray();
        return new PtNDArray(
                ndArray.getManager(),
                PyTorchLibrary.LIB.torchRot90(ndArray.getHandle(), times, longaxes));
    }

    public static PtNDArray sum(PtNDArray ndArray) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchSum(ndArray.getHandle()));
    }

    public static PtNDArray sum(PtNDArray ndArray, long[] dims, boolean keepDim) {
        return new PtNDArray(
                ndArray.getManager(),
                PyTorchLibrary.LIB.torchSum(ndArray.getHandle(), dims, keepDim));
    }

    public static PtNDArray prod(PtNDArray ndArray) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchProd(ndArray.getHandle()));
    }

    public static PtNDArray prod(PtNDArray ndArray, long dim, boolean keepDim) {
        return new PtNDArray(
                ndArray.getManager(),
                PyTorchLibrary.LIB.torchProd(ndArray.getHandle(), dim, keepDim));
    }

    public static PtNDArray cumSum(PtNDArray ndArray, long dim) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchCumSum(ndArray.getHandle(), dim));
    }

    public static PtNDArray oneHot(PtNDArray ndArray, int depth, DataType dataType) {
        return new PtNDArray(
                        ndArray.getManager(),
                        PyTorchLibrary.LIB.torchNNOneHot(
                                ndArray.toType(DataType.INT64, false).getHandle(), depth))
                .toType(dataType, false);
    }

    public static NDList split(PtNDArray ndArray, long size, long axis) {
        long[] ndPtrs = PyTorchLibrary.LIB.torchSplit(ndArray.getHandle(), size, axis);
        NDList list = new NDList();
        for (long ptr : ndPtrs) {
            list.add(new PtNDArray(ndArray.getManager(), ptr));
        }
        return list;
    }

    public static NDList split(PtNDArray ndArray, long[] indices, long axis) {
        long[] ndPtrs = PyTorchLibrary.LIB.torchSplit(ndArray.getHandle(), indices, axis);
        NDList list = new NDList();
        for (long ptr : ndPtrs) {
            list.add(new PtNDArray(ndArray.getManager(), ptr));
        }
        return list;
    }

    public static PtNDArray squeeze(PtNDArray ndArray) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchSqueeze(ndArray.getHandle()));
    }

    public static PtNDArray squeeze(PtNDArray ndArray, long dim) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchSqueeze(ndArray.getHandle(), dim));
    }

    public static PtNDArray unsqueeze(PtNDArray ndArray, long dim) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchUnsqueeze(ndArray.getHandle(), dim));
    }

    public static PtNDArray flatten(PtNDArray ndArray, long startDim, long endDim) {
        return new PtNDArray(
                ndArray.getManager(),
                PyTorchLibrary.LIB.torchFlatten(ndArray.getHandle(), startDim, endDim));
    }

    public static PtNDArray abs(PtNDArray ndArray) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchAbs(ndArray.getHandle()));
    }

    public static PtNDArray square(PtNDArray ndArray) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchSquare(ndArray.getHandle()));
    }

    public static PtNDArray floor(PtNDArray ndArray) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchFloor(ndArray.getHandle()));
    }

    public static PtNDArray ceil(PtNDArray ndArray) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchCeil(ndArray.getHandle()));
    }

    public static PtNDArray round(PtNDArray ndArray) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchRound(ndArray.getHandle()));
    }

    public static PtNDArray trunc(PtNDArray ndArray) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchTrunc(ndArray.getHandle()));
    }

    public static PtNDArray clip(PtNDArray ndArray, Number min, Number max) {
        PtNDArray minNd = (PtNDArray) ndArray.getManager().create(min);
        PtNDArray maxNd = (PtNDArray) ndArray.getManager().create(max);
        return new PtNDArray(
                ndArray.getManager(),
                PyTorchLibrary.LIB.torchClamp(
                        ndArray.getHandle(), minNd.getHandle(), maxNd.getHandle()));
    }

    public static PtNDArray exp(PtNDArray ndArray) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchExp(ndArray.getHandle()));
    }

    public static PtNDArray log(PtNDArray ndArray) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchLog(ndArray.getHandle()));
    }

    public static PtNDArray log10(PtNDArray ndArray) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchLog10(ndArray.getHandle()));
    }

    public static PtNDArray log2(PtNDArray ndArray) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchLog2(ndArray.getHandle()));
    }

    public static PtNDArray sin(PtNDArray ndArray) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchSin(ndArray.getHandle()));
    }

    public static PtNDArray cos(PtNDArray ndArray) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchCos(ndArray.getHandle()));
    }

    public static PtNDArray tan(PtNDArray ndArray) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchTan(ndArray.getHandle()));
    }

    public static PtNDArray asin(PtNDArray ndArray) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchASin(ndArray.getHandle()));
    }

    public static PtNDArray acos(PtNDArray ndArray) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchAcos(ndArray.getHandle()));
    }

    public static PtNDArray atan(PtNDArray ndArray) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchAtan(ndArray.getHandle()));
    }

    public static PtNDArray sqrt(PtNDArray ndArray) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchSqrt(ndArray.getHandle()));
    }

    public static PtNDArray sinh(PtNDArray ndArray) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchSinh(ndArray.getHandle()));
    }

    public static PtNDArray cosh(PtNDArray ndArray) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchCosh(ndArray.getHandle()));
    }

    public static PtNDArray tanh(PtNDArray ndArray) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchTanh(ndArray.getHandle()));
    }

    public static PtNDArray sigmoid(PtNDArray ndArray) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchSigmoid(ndArray.getHandle()));
    }

    public static PtNDArray all(PtNDArray ndArray) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchAll(ndArray.getHandle()));
    }

    public static PtNDArray any(PtNDArray ndArray) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchAny(ndArray.getHandle()));
    }

    public static PtNDArray none(PtNDArray ndArray) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchNone(ndArray.getHandle()));
    }

    public static PtNDArray eq(PtNDArray self, PtNDArray other) {
        return new PtNDArray(
                self.getManager(), PyTorchLibrary.LIB.torchEq(self.getHandle(), other.getHandle()));
    }

    public static PtNDArray neq(PtNDArray self, PtNDArray other) {
        return new PtNDArray(
                self.getManager(),
                PyTorchLibrary.LIB.torchNeq(self.getHandle(), other.getHandle()));
    }

    public static PtNDArray gt(PtNDArray self, PtNDArray other) {
        return new PtNDArray(
                self.getManager(), PyTorchLibrary.LIB.torchGt(self.getHandle(), other.getHandle()));
    }

    public static PtNDArray gte(PtNDArray self, PtNDArray other) {
        return new PtNDArray(
                self.getManager(),
                PyTorchLibrary.LIB.torchGte(self.getHandle(), other.getHandle()));
    }

    public static PtNDArray lt(PtNDArray self, PtNDArray other) {
        return new PtNDArray(
                self.getManager(), PyTorchLibrary.LIB.torchLt(self.getHandle(), other.getHandle()));
    }

    public static PtNDArray lte(PtNDArray self, PtNDArray other) {
        return new PtNDArray(
                self.getManager(),
                PyTorchLibrary.LIB.torchLte(self.getHandle(), other.getHandle()));
    }

    public static PtNDArray neg(PtNDArray ndArray) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchNeg(ndArray.getHandle()));
    }

    public static void negi(PtNDArray ndArray) {
        PyTorchLibrary.LIB.torchNegi(ndArray.getHandle());
    }

    public static PtNDArray isNaN(PtNDArray ndArray) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchIsNaN(ndArray.getHandle()));
    }

    public static PtNDArray isInf(PtNDArray ndArray) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchIsInf(ndArray.getHandle()));
    }

    public static PtNDArray randint(
            PtNDManager manager,
            long low,
            long high,
            Shape size,
            DataType dataType,
            Device device) {
        return new PtNDArray(
                manager,
                PyTorchLibrary.LIB.torchRandint(
                        low,
                        high,
                        size.getShape(),
                        dataType.ordinal(),
                        layoutMapper(SparseFormat.DENSE, device),
                        new int[] {PtDeviceType.toDeviceType(device), device.getDeviceId()},
                        false));
    }

    public static PtNDArray normal(
            PtNDManager manager,
            double mean,
            double std,
            Shape size,
            DataType dataType,
            Device device) {
        return new PtNDArray(
                manager,
                PyTorchLibrary.LIB.torchNormal(
                        mean,
                        std,
                        size.getShape(),
                        dataType.ordinal(),
                        layoutMapper(SparseFormat.DENSE, device),
                        new int[] {PtDeviceType.toDeviceType(device), device.getDeviceId()},
                        false));
    }

    public static PtNDArray uniform(
            PtNDManager manager,
            double low,
            double high,
            Shape size,
            DataType dataType,
            Device device) {
        return new PtNDArray(
                manager,
                PyTorchLibrary.LIB.tensorUniform(
                        low,
                        high,
                        size.getShape(),
                        dataType.ordinal(),
                        layoutMapper(SparseFormat.DENSE, device),
                        new int[] {PtDeviceType.toDeviceType(device), device.getDeviceId()},
                        false));
    }

    public static PtNDArray eye(
            PtNDManager manager, int n, int m, DataType dataType, Device device, SparseFormat fmt) {
        return new PtNDArray(
                manager,
                PyTorchLibrary.LIB.torchEye(
                        n,
                        m,
                        dataType.ordinal(),
                        layoutMapper(fmt, device),
                        new int[] {PtDeviceType.toDeviceType(device), device.getDeviceId()},
                        false));
    }

    public static PtNDArray erfinv(PtNDArray ndArray) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchErfinv(ndArray.getHandle()));
    }

    public static PtNDArray interpolate(
            PtNDArray ndArray, long[] size, int mode, boolean alignCorners) {
        return new PtNDArray(
                ndArray.getManager(),
                PyTorchLibrary.LIB.torchNNInterpolate(
                        ndArray.getHandle(), size, mode, alignCorners));
    }

    public static PtNDArray linear(PtNDArray input, PtNDArray weight, PtNDArray bias) {
        return new PtNDArray(
                input.getManager(),
                PyTorchLibrary.LIB.torchNNLinear(
                        input.getHandle(),
                        weight.getHandle(),
                        bias == null ? NULL_PTR : bias.getHandle()));
    }

    public static PtNDArray embedding(PtNDArray input, PtNDArray weight, boolean sparse) {
        return new PtNDArray(
                input.getManager(),
                PyTorchLibrary.LIB.torchNNEmbedding(input.getHandle(), weight.getHandle(), sparse));
    }

    public static PtNDArray relu(PtNDArray ndArray) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchNNRelu(ndArray.getHandle()));
    }

    public static PtNDArray softPlus(PtNDArray ndArray) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchNNSoftPlus(ndArray.getHandle()));
    }

    public static PtNDArray softSign(PtNDArray ndArray) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchNNSoftSign(ndArray.getHandle()));
    }

    public static PtNDArray leakyRelu(PtNDArray ndArray, double negativeSlope) {
        return new PtNDArray(
                ndArray.getManager(),
                PyTorchLibrary.LIB.torchNNLeakyRelu(ndArray.getHandle(), negativeSlope));
    }

    public static PtNDArray elu(PtNDArray ndArray, double alpha) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchNNElu(ndArray.getHandle(), alpha));
    }

    public static PtNDArray selu(PtNDArray ndArray) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchNNSelu(ndArray.getHandle()));
    }

    public static PtNDArray gelu(PtNDArray ndArray) {
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchNNGelu(ndArray.getHandle()));
    }

    public static PtNDArray convolution(
            PtNDArray ndArray,
            PtNDArray weight,
            PtNDArray bias,
            Shape stride,
            Shape padding,
            Shape dilation,
            int groups) {
        return new PtNDArray(
                ndArray.getManager(),
                PyTorchLibrary.LIB.torchNNConvNd(
                        ndArray.getHandle(),
                        weight.getHandle(),
                        (bias != null) ? bias.getHandle() : NULL_PTR,
                        stride.getShape(),
                        padding.getShape(),
                        dilation.getShape(),
                        groups));
    }

    public static PtNDArray batchNorm(
            PtNDArray ndArray,
            PtNDArray gamma,
            PtNDArray beta,
            PtNDArray runningMean,
            PtNDArray runningVar,
            boolean isTraining,
            double momentum,
            double eps) {
        return new PtNDArray(
                ndArray.getManager(),
                PyTorchLibrary.LIB.torchNNBatchNorm(
                        ndArray.getHandle(),
                        gamma.getHandle(),
                        beta.getHandle(),
                        runningMean.getHandle(),
                        runningVar.getHandle(),
                        isTraining,
                        momentum,
                        eps));
    }

    public static PtNDArray layerNorm(
            PtNDArray ndArray, Shape normalizedShape, PtNDArray gamma, PtNDArray beta, double eps) {
        return new PtNDArray(
                ndArray.getManager(),
                PyTorchLibrary.LIB.torchNNLayerNorm(
                        ndArray.getHandle(),
                        normalizedShape.getShape(),
                        gamma.getHandle(),
                        beta.getHandle(),
                        eps));
    }

    public static PtNDArray dropout(PtNDArray ndArray, double prob, boolean training) {
        return new PtNDArray(
                ndArray.getManager(),
                PyTorchLibrary.LIB.torchNNDropout(ndArray.getHandle(), prob, training));
    }

    public static NDList rnn(
            PtNDArray input,
            PtNDArray hx,
            NDList params,
            boolean hasBiases,
            int numLayers,
            RNN.Activation activation,
            double dropRate,
            boolean training,
            boolean bidirectional,
            boolean batchFirst) {
        PtNDManager manager = input.getManager();
        long[] paramHandles =
                params.stream().mapToLong(array -> ((PtNDArray) array).getHandle()).toArray();
        long[] outputs =
                PyTorchLibrary.LIB.torchNNRnn(
                        input.getHandle(),
                        hx.getHandle(),
                        paramHandles,
                        hasBiases,
                        numLayers,
                        activation.ordinal(),
                        dropRate,
                        training,
                        bidirectional,
                        batchFirst);
        NDList res = new NDList();
        for (long output : outputs) {
            res.add(new PtNDArray(manager, output));
        }
        return res;
    }

    public static NDList gru(
            PtNDArray input,
            PtNDArray hx,
            NDList params,
            boolean hasBiases,
            int numLayers,
            double dropRate,
            boolean training,
            boolean bidirectional,
            boolean batchFirst) {
        PtNDManager manager = input.getManager();
        long[] paramHandles =
                params.stream().mapToLong(array -> ((PtNDArray) array).getHandle()).toArray();
        long[] outputs =
                PyTorchLibrary.LIB.torchNNGru(
                        input.getHandle(),
                        hx.getHandle(),
                        paramHandles,
                        hasBiases,
                        numLayers,
                        dropRate,
                        training,
                        bidirectional,
                        batchFirst);
        NDList res = new NDList();
        for (long output : outputs) {
            res.add(new PtNDArray(manager, output));
        }
        return res;
    }

    public static NDList lstm(
            PtNDArray input,
            NDList hx,
            NDList params,
            boolean hasBiases,
            int numLayers,
            double dropRate,
            boolean training,
            boolean bidirectional,
            boolean batchFirst) {
        PtNDManager manager = input.getManager();
        long[] hxHandles =
                hx.stream().mapToLong(array -> ((PtNDArray) array).getHandle()).toArray();
        long[] paramHandles =
                params.stream().mapToLong(array -> ((PtNDArray) array).getHandle()).toArray();
        long[] outputs =
                PyTorchLibrary.LIB.torchNNLstm(
                        input.getHandle(),
                        hxHandles,
                        paramHandles,
                        hasBiases,
                        numLayers,
                        dropRate,
                        training,
                        bidirectional,
                        batchFirst);
        NDList res = new NDList();
        for (long output : outputs) {
            res.add(new PtNDArray(manager, output));
        }
        return res;
    }

    public static PtNDArray avgPool(
            PtNDArray ndArray,
            Shape kernelSize,
            Shape stride,
            Shape padding,
            boolean ceilMode,
            boolean countIncludePad) {
        return new PtNDArray(
                ndArray.getManager(),
                PyTorchLibrary.LIB.torchNNAvgPool(
                        ndArray.getHandle(),
                        kernelSize.getShape(),
                        stride.getShape(),
                        padding.getShape(),
                        ceilMode,
                        countIncludePad));
    }

    public static PtNDArray maxPool(
            PtNDArray ndArray, Shape kernelSize, Shape stride, Shape padding, boolean ceilMode) {
        return new PtNDArray(
                ndArray.getManager(),
                PyTorchLibrary.LIB.torchNNMaxPool(
                        ndArray.getHandle(),
                        kernelSize.getShape(),
                        stride.getShape(),
                        padding.getShape(),
                        ceilMode));
    }

    public static PtNDArray adaptiveMaxPool(PtNDArray ndArray, Shape outputSize) {
        return new PtNDArray(
                ndArray.getManager(),
                PyTorchLibrary.LIB.torchNNAdaptiveMaxPool(
                        ndArray.getHandle(), outputSize.getShape()));
    }

    public static PtNDArray adaptiveAvgPool(PtNDArray ndArray, Shape outputSize) {
        return new PtNDArray(
                ndArray.getManager(),
                PyTorchLibrary.LIB.torchNNAdaptiveAvgPool(
                        ndArray.getHandle(), outputSize.getShape()));
    }

    public static PtNDArray lpPool(
            PtNDArray ndArray, double normType, Shape kernelSize, Shape stride, boolean ceilMode) {
        if (ndArray.getShape().dimension() - 2 == 3) {
            throw new UnsupportedOperationException("3D lpPool is not supported in PyTorch engine");
        }
        return new PtNDArray(
                ndArray.getManager(),
                PyTorchLibrary.LIB.torchNNLpPool(
                        ndArray.getHandle(),
                        normType,
                        kernelSize.getShape(),
                        stride.getShape(),
                        ceilMode));
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
        } else if (layout == 2) {
            logger.debug("MKLDNN layout is used!");
            return SparseFormat.DENSE;
        }
        throw new UnsupportedOperationException("Unsupported data format");
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

    public static void deleteNDArray(long handle) {
        PyTorchLibrary.LIB.torchDeleteTensor(handle);
    }

    public static boolean requiresGrad(PtNDArray ndArray) {
        return PyTorchLibrary.LIB.torchRequiresGrad(ndArray.getHandle());
    }

    public static String getGradientFunctionNames(PtNDArray ndArray) {
        return PyTorchLibrary.LIB.torchGradFnName(ndArray.getHandle());
    }

    public static void attachGradient(PtNDArray ndArray, boolean requiresGrad) {
        PyTorchLibrary.LIB.torchAttachGrad(ndArray.getHandle(), requiresGrad);
    }

    public static PtNDArray detachGradient(PtNDArray ndArray) {
        // TODO: detached ndarray may not use the same manager for the attached one
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchDetachGrad(ndArray.getHandle()));
    }

    public static PtNDArray getGradient(PtNDArray ndArray) {
        long pointer = PyTorchLibrary.LIB.torchGrad(ndArray.getHandle());
        if (pointer == NULL_PTR) {
            return null;
        }
        return new PtNDArray(ndArray.getManager(), pointer);
    }

    public static void backward(
            PtNDArray ndArray, PtNDArray gradNd, boolean keepGraph, boolean createGraph) {
        PyTorchLibrary.LIB.torchBackward(
                ndArray.getHandle(), gradNd.getHandle(), keepGraph, createGraph);
    }

    public static void deleteModule(long pointer) {
        PyTorchLibrary.LIB.torchDeleteModule(pointer);
    }

    public static void setGraphExecutorOptimize(boolean enabled) {
        PyTorchLibrary.LIB.setGraphExecutorOptimize(enabled);
    }

    public static PtSymbolBlock loadModule(
            PtNDManager manager,
            Path path,
            Device device,
            String[] extraFileKeys,
            String[] extraFileValues) {
        long handle =
                PyTorchLibrary.LIB.moduleLoad(
                        path.toString(),
                        new int[] {PtDeviceType.toDeviceType(device), device.getDeviceId()},
                        extraFileKeys,
                        extraFileValues);
        return new PtSymbolBlock(manager, handle);
    }

    public static PtSymbolBlock loadModule(
            PtNDManager manager, InputStream is, Device device, boolean hasSize)
            throws IOException {
        long handle = loadModuleHandle(is, device, hasSize);
        return new PtSymbolBlock(manager, handle);
    }

    public static long loadModuleHandle(InputStream is, Device device, boolean hasSize)
            throws IOException {
        byte[] buf = new byte[BYTE_LENGTH];
        long size = -1;
        if (hasSize) {
            size = new DataInputStream(is).readLong();
        }
        return PyTorchLibrary.LIB.moduleLoad(
                is, new int[] {PtDeviceType.toDeviceType(device), device.getDeviceId()}, buf, size);
    }

    public static void writeModule(PtSymbolBlock block, OutputStream os, boolean writeSize) {
        byte[] buf = new byte[BYTE_LENGTH];
        PyTorchLibrary.LIB.moduleWrite(block.getHandle(), os, buf, writeSize);
    }

    public static NDList moduleGetParams(PtSymbolBlock block, PtNDManager manager) {
        long[] handles = PyTorchLibrary.LIB.moduleGetParams(block.getHandle());
        String[] names = PyTorchLibrary.LIB.moduleGetParamNames(block.getHandle());
        NDList list = new NDList(handles.length);
        for (int i = 0; i < handles.length; i++) {
            PtNDArray array = new PtNDArray(manager, handles[i]);
            array.setName(names[i]);
            list.add(array);
        }
        return list;
    }

    public static void enableInferenceMode(PtSymbolBlock block) {
        PyTorchLibrary.LIB.moduleEval(block.getHandle());
    }

    public static void enableTrainingMode(PtSymbolBlock block) {
        PyTorchLibrary.LIB.moduleTrain(block.getHandle());
    }

    public static void zeroGrad(PtNDArray weight) {
        PyTorchLibrary.LIB.zeroGrad(weight.getHandle());
    }

    public static void adamUpdate(
            PtNDArray weight,
            PtNDArray grad,
            PtNDArray mean,
            PtNDArray variance,
            float lr,
            float wd,
            float rescaleGrad,
            float clipGrad,
            float beta1,
            float beta2,
            float eps) {
        PyTorchLibrary.LIB.adamUpdate(
                weight.getHandle(),
                grad.getHandle(),
                mean.getHandle(),
                variance.getHandle(),
                lr,
                wd,
                rescaleGrad,
                clipGrad,
                beta1,
                beta2,
                eps);
    }

    public static void sgdUpdate(
            PtNDArray weight,
            PtNDArray grad,
            PtNDArray state,
            float lr,
            float wd,
            float rescaleGrad,
            float clipGrad,
            float momentum) {
        PyTorchLibrary.LIB.sgdUpdate(
                weight.getHandle(),
                grad.getHandle(),
                (state == null) ? NULL_PTR : state.getHandle(),
                lr,
                wd,
                rescaleGrad,
                clipGrad,
                momentum);
    }

    // Internal use only
    public static int getLayout(PtNDArray array) {
        return PyTorchLibrary.LIB.torchLayout(array.getHandle());
    }

    public static PtNDArray norm(PtNDArray ndArray, int ord, int[] axes, boolean keepDims) {
        long[] longAxes = Arrays.stream(axes).mapToLong(i -> i).toArray();
        return new PtNDArray(
                ndArray.getManager(),
                PyTorchLibrary.LIB.torchNorm(ndArray.getHandle(), ord, longAxes, keepDims));
    }

    public static PtNDArray nonZeros(PtNDArray ndArray) {
        if (ndArray.isScalar()) {
            ndArray = (PtNDArray) ndArray.reshape(-1);
        }
        return new PtNDArray(
                ndArray.getManager(), PyTorchLibrary.LIB.torchNonZeros(ndArray.getHandle()));
    }
}
