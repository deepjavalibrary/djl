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
package ai.djl.paddlepaddle.jni;

import ai.djl.Device;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.paddlepaddle.engine.PaddlePredictor;
import ai.djl.paddlepaddle.engine.PpDataType;
import ai.djl.paddlepaddle.engine.PpNDArray;
import ai.djl.paddlepaddle.engine.PpNDManager;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;

/**
 * A class containing utilities to interact with the Paddle Engine's Java Native Interface (JNI)
 * layer.
 */
@SuppressWarnings("MissingJavadocMethod")
public final class JniUtils {
    private JniUtils() {}

    private static String[] libArgs;

    public static void loadExtraDir(String[] args) {
        if (args != null) {
            libArgs = args;
        }
        PaddleLibrary.LIB.loadExtraDir(libArgs);
    }

    public static PpNDArray createNdArray(
            PpNDManager manager, ByteBuffer data, Shape shape, DataType dtype) {
        int[] intShape = Arrays.stream(shape.getShape()).mapToInt(Math::toIntExact).toArray();
        long handle =
                PaddleLibrary.LIB.paddleCreateTensor(
                        data, data.remaining(), intShape, PpDataType.toPaddlePaddle(dtype));
        return new PpNDArray(manager, handle);
    }

    public static DataType getDTypeFromNd(PpNDArray array) {
        int type = PaddleLibrary.LIB.getTensorDType(array.getHandle());
        return PpDataType.fromPaddlePaddle(type);
    }

    public static ByteBuffer getByteBufferFromNd(PpNDArray array) {
        ByteBuffer bb = ByteBuffer.wrap(PaddleLibrary.LIB.getTensorData(array.getHandle()));
        return bb.order(ByteOrder.nativeOrder());
    }

    public static Shape getShapeFromNd(PpNDArray array) {
        int[] shape = PaddleLibrary.LIB.getTensorShape(array.getHandle());
        return new Shape(Arrays.stream(shape).asLongStream().toArray());
    }

    public static void setNdName(PpNDArray array, String name) {
        PaddleLibrary.LIB.setTensorName(array.getHandle(), name);
    }

    public static String getNameFromNd(PpNDArray array) {
        return PaddleLibrary.LIB.getTensorName(array.getHandle());
    }

    public static void deleteNd(Long handle) {
        PaddleLibrary.LIB.deleteTensor(handle);
    }

    public static long createConfig(String modelDir, String paramDir, Device device) {
        int deviceId = device == Device.cpu() ? -1 : device.getDeviceId();
        return PaddleLibrary.LIB.createAnalysisConfig(modelDir, paramDir, deviceId);
    }

    public static void useFeedFetchOp(long config) {
        PaddleLibrary.LIB.useFeedFetchOp(config);
    }

    public static void deleteConfig(long config) {
        PaddleLibrary.LIB.deleteAnalysisConfig(config);
    }

    public static long createPredictor(long config) {
        loadExtraDir(null);
        return PaddleLibrary.LIB.createPredictor(config);
    }

    public static long clonePredictor(PaddlePredictor predictor) {
        return PaddleLibrary.LIB.clonePredictor(predictor.getHandle());
    }

    public static void deletePredictor(PaddlePredictor predictor) {
        PaddleLibrary.LIB.deletePredictor(predictor.getHandle());
    }

    public static PpNDArray[] predictorForward(
            PaddlePredictor predictor, PpNDArray[] inputs, String[] inputNames) {
        long[] inputHandles = new long[inputs.length];
        for (int i = 0; i < inputs.length; i++) {
            inputs[i].setName(inputNames[i]);
            inputHandles[i] = inputs[i].getHandle();
        }
        long[] outputs = PaddleLibrary.LIB.runInference(predictor.getHandle(), inputHandles);
        PpNDManager manager = (PpNDManager) inputs[0].getManager();
        PpNDArray[] arrays = new PpNDArray[outputs.length];
        for (int i = 0; i < outputs.length; i++) {
            arrays[i] = new PpNDArray(manager, outputs[i]);
        }
        return arrays;
    }

    public static String[] getInputNames(PaddlePredictor predictor) {
        return PaddleLibrary.LIB.getInputNames(predictor.getHandle());
    }

    public static String getVersion() {
        return "2.0.0";
    }
}
