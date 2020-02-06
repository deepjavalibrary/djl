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

import java.nio.ByteBuffer;

/** A class containing utilities to interact with the PyTorch Engine's JNI layer. */
@SuppressWarnings("MissingJavadocMethod")
public final class PyTorchLibrary {

    static {
        System.loadLibrary("djl_torch"); // NOPMD
    }

    public static final PyTorchLibrary LIB = new PyTorchLibrary();

    private PyTorchLibrary() {}

    public native long[] torchSizes(Pointer handle);

    public native ByteBuffer torchDataPtr(Pointer handle);

    public native int torchDType(Pointer handle);

    public native int[] torchDevice(Pointer handle);

    public native int torchLayout(Pointer handle);

    public native Pointer torchTo(Pointer handle, int dType, int[] device, boolean copy);

    public native Pointer torchEmpty(
            long[] shape, int dType, int layout, int[] device, boolean requiredGrad);

    public native Pointer torchZeros(
            long[] shape, int dType, int layout, int[] device, boolean requiredGrad);

    public native Pointer torchOnes(
            long[] shape, int dType, int layout, int[] device, boolean requiredGrad);

    public native Pointer torchArange(
            int start,
            int end,
            int step,
            int dType,
            int layout,
            int[] device,
            boolean requiredGrad);

    public native Pointer torchArange(
            double start,
            double end,
            double step,
            int dType,
            int layout,
            int[] device,
            boolean requiredGrad);

    public native Pointer torchSub(Pointer handle, double scalar);

    public native Pointer torchDiv(Pointer handle, double scalar);

    public native Pointer torchReshape(Pointer handle, long[] shape);

    public native Pointer torchSoftmax(Pointer handle, long dim, int dType);

    public native Pointer torchArgMax(Pointer handle);

    public native Pointer torchArgMax(Pointer handle, long dim, boolean keepDim);

    public native boolean contentEqual(Pointer handle1, Pointer handle2);

    public native Pointer torchFromBlob(
            ByteBuffer data,
            long[] shape,
            int dType,
            int layout,
            int[] device,
            boolean requiredGrad);

    public native Pointer torchGet(Pointer handle, long dim, long start);

    public native void torchDeleteTensor(Pointer handle);

    public native void torchDeleteModule(Pointer handle);

    public native Pointer[] torchSplit(Pointer handle, long size, long dim);

    public native Pointer[] torchSplit(Pointer handle, long[] indices, long dim);

    public native Pointer torchStack(Pointer[] handles, long dim);

    public native Pointer torchAbs(Pointer handle);

    public native Pointer torchFloor(Pointer handle);

    public native Pointer torchCeil(Pointer handle);

    public native Pointer torchRound(Pointer handle);

    public native Pointer torchTrunc(Pointer handle);

    public native Pointer torchExp(Pointer handle);

    public native Pointer torchLog(Pointer handle);

    public native Pointer torchLog10(Pointer handle);

    public native Pointer torchLog2(Pointer handle);

    public native Pointer torchSin(Pointer handle);

    public native Pointer torchCos(Pointer handle);

    public native Pointer torchTan(Pointer handle);

    public native Pointer torchASin(Pointer handle);

    public native Pointer torchAcos(Pointer handle);

    public native Pointer torchAtan(Pointer handle);

    public native Pointer torchSqrt(Pointer handle);

    public native Pointer torchSinh(Pointer handle);

    public native Pointer torchCosh(Pointer handle);

    public native Pointer torchTanh(Pointer handle);

    public native Pointer torchEq(Pointer self, Pointer other);

    public native Pointer torchNeq(Pointer self, Pointer other);

    public native Pointer torchGt(Pointer self, Pointer other);

    public native Pointer torchGte(Pointer self, Pointer other);

    public native Pointer torchLt(Pointer self, Pointer other);

    public native Pointer torchLte(Pointer self, Pointer other);

    public native Pointer normalize(Pointer handle, float[] mean, float[] std);

    public native Pointer resize(Pointer handle, long[] size, boolean alignCorners);

    public native Pointer moduleLoad(String path);

    public native void moduleEval(Pointer handle);

    public native Pointer moduleForward(Pointer moduleHandle, Pointer[] iValuePointers);

    public native Pointer iValueCreateFromTensor(Pointer tensorHandle);

    public native Pointer iValueToTensor(Pointer iValueHandle);
}
