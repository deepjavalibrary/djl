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
package ai.djl.paddlepaddle.jna;

import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.paddlepaddle.engine.PpDataType;
import ai.djl.paddlepaddle.engine.PpNDArray;
import ai.djl.paddlepaddle.engine.PpNDManager;
import ai.djl.util.NativeResource;
import com.sun.jna.Memory;
import com.sun.jna.Native;
import com.sun.jna.Pointer;
import com.sun.jna.Structure;
import com.sun.jna.ptr.PointerByReference;

import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.IntBuffer;
import java.util.Arrays;

/**
 * A class containing utilities to interact with the PaddlePaddle Engine's Java Native Access (JNA)
 * layer.
 */
@SuppressWarnings("MissingJavadocMethod")
public final class JnaUtils {

    private static final PaddleLibrary LIB = LibUtils.loadLibrary();

    private JnaUtils() {}

    public static PpNDArray createNdArray(
            PpNDManager manager, Buffer data, Shape shape, DataType dtype) {
        Pointer tensor = LIB.PD_NewPaddleTensor();
        LIB.PD_SetPaddleTensorDType(tensor, PpDataType.toPaddlePaddle(dtype));
        long[] shapes = shape.getShape();
        int[] size = Arrays.stream(shapes).mapToInt(Math::toIntExact).toArray();
        LIB.PD_SetPaddleTensorShape(tensor, size, size.length);

        Pointer paddleBuffer = LIB.PD_NewPaddleBuf();
        long length = dtype.getNumOfBytes() * shape.size();
        Pointer pointer = Native.getDirectBufferPointer(data);
        LIB.PD_PaddleBufReset(paddleBuffer, pointer, length);
        LIB.PD_SetPaddleTensorData(tensor, paddleBuffer);
        return new PpNDArray(manager, tensor, shape, dtype);
    }

    public static Pointer getBufferPointerFromNd(PpNDArray array) {
        Pointer bufHandle = LIB.PD_GetPaddleTensorData(array.getHandle());
        return LIB.PD_PaddleBufData(bufHandle);
    }

    public static ByteBuffer getByteBufferFromNd(PpNDArray array) {
        Pointer bufHandle = LIB.PD_GetPaddleTensorData(array.getHandle());
        int length = Math.toIntExact(LIB.PD_PaddleBufLength(bufHandle));
        Pointer buf = LIB.PD_PaddleBufData(bufHandle);
        byte[] bytes = new byte[length];
        buf.read(0, bytes, 0, length);
        return ByteBuffer.wrap(bytes).order(ByteOrder.nativeOrder());
    }

    public static void freeNdArray(Pointer tensor) {
        LIB.PD_DeletePaddleTensor(tensor);
    }

    public static void setNdArrayName(PpNDArray nd, String name) {
        LIB.PD_SetPaddleTensorName(nd.getHandle(), name);
    }

    public static String getNdArrayName(PpNDArray nd) {
        return LIB.PD_GetPaddleTensorName(nd.getHandle());
    }

    public static DataType getDataType(PpNDArray nd) {
        int type = LIB.PD_GetPaddleTensorDType(nd.getHandle());
        return PpDataType.fromPaddlePaddle(type);
    }

    public static Shape getShape(PpNDArray nd) {
        IntBuffer outSize = IntBuffer.allocate(1);
        Pointer shapePtr = LIB.PD_GetPaddleTensorShape(nd.getHandle(), outSize);
        int[] shape = shapePtr.getIntArray(0, outSize.get());
        return new Shape(Arrays.stream(shape).asLongStream().toArray());
    }

    public static Pointer newAnalysisConfig() {
        return LIB.PD_NewAnalysisConfig();
    }

    public static void disableGpu(AnalysisConfig config) {
        LIB.PD_DisableGpu(config.getHandle());
    }

    public static void setGpu(AnalysisConfig config, int memory, int deviceId) {
        LIB.PD_EnableUseGpu(config.getHandle(), memory, deviceId);
    }

    public static void loadModel(AnalysisConfig config, String modelDir, String paramsPath) {
        if (paramsPath == null) {
            paramsPath = modelDir;
        }
        LIB.PD_SetModel(config.getHandle(), modelDir, paramsPath);
    }

    public static void deleteConfig(AnalysisConfig config) {
        LIB.PD_DeleteAnalysisConfig(config.getHandle());
    }

    public static PpNDArray[] runInference(AnalysisConfig config, PpNDArray[] inputs, int batchSize) {
        PointerArray inputPtr = new PointerArray(Arrays.stream(inputs).map(PpNDArray::getHandle).toArray(Pointer[]::new));
        PointerByReference outputPtr = new PointerByReference();
        IntBuffer outSizeBuf = IntBuffer.allocate(1);
        LIB.PD_PredictorRun(config.getHandle(), inputPtr, inputs.length, outputPtr, outSizeBuf, batchSize);
        Pointer[] handles = outputPtr.getValue().getPointerArray(0, outSizeBuf.get());
        PpNDManager manager = (PpNDManager) inputs[0].getManager();
        return Arrays.stream(handles).map(ptr -> new PpNDArray(manager, ptr)).toArray(PpNDArray[]::new);
    }

    public static String getVersion() {
        return "2.0.0";
    }
}
