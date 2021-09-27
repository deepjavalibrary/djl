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
package ai.djl.dlr.jni;

import ai.djl.Device;
import ai.djl.dlr.engine.DlrNDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;

/**
 * A class containing utilities to interact with the PyTorch Engine's Java Native Interface (JNI)
 * layer.
 */
@SuppressWarnings("MissingJavadocMethod")
public final class JniUtils {

    private JniUtils() {}

    public static void setDlrInput(long modelHandle, DlrNDArray input, int index) {
        long[] shape = input.getShape().getShape();
        float[] data = input.toFloatArray();
        String name = DlrLibrary.LIB.getDlrInputName(modelHandle, index);
        DlrLibrary.LIB.setDLRInput(modelHandle, name, shape, data, shape.length);
    }

    public static NDList getDlrOutputs(long modelHandle, NDManager manager) {
        int numOutputs = DlrLibrary.LIB.getDlrNumOutputs(modelHandle);
        NDList res = new NDList(numOutputs);
        for (int i = 0; i < numOutputs; i++) {
            float[] data = DlrLibrary.LIB.getDlrOutput(modelHandle, i);
            long[] shape = DlrLibrary.LIB.getDlrOutputShape(modelHandle, i);
            res.add(manager.create(data, new Shape(shape)));
        }
        return res;
    }

    public static long createDlrModel(String path, Device device) {
        int deviceId = 0;
        if (!device.equals(Device.cpu())) {
            deviceId = device.getDeviceId();
        }
        return DlrLibrary.LIB.createDlrModel(path, mapDevice(device.getDeviceType()), deviceId);
    }

    public static void deleteDlrModel(long modelHandle) {
        DlrLibrary.LIB.deleteDlrModel(modelHandle);
    }

    public static void runDlrModel(long modelHandle) {
        DlrLibrary.LIB.runDlrModel(modelHandle);
    }

    public static void setDlrNumThreads(long modelHandle, int threads) {
        DlrLibrary.LIB.setDlrNumThreads(modelHandle, threads);
    }

    public static void useDlrCpuAffinity(long modelHandle, boolean use) {
        DlrLibrary.LIB.useDlrCPUAffinity(modelHandle, use);
    }

    public static String getDlrVersion() {
        return DlrLibrary.LIB.getDlrVersion();
    }

    private static int mapDevice(String deviceType) {
        if (Device.Type.CPU.equals(deviceType)) {
            return 1;
        } else if (Device.Type.GPU.equals(deviceType)) {
            return 2;
        } else {
            throw new IllegalArgumentException("The device " + deviceType + " is not supported");
        }
    }
}
