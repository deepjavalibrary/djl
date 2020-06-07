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
import java.util.Set;

/** A class containing utilities to interact with the PyTorch Engine's JNI layer. */
final class PyTorchLibrary {

    static final PyTorchLibrary LIB = new PyTorchLibrary();

    private PyTorchLibrary() {}

    native void torchSetNumInteropThreads(int threads);

    native void torchSetNumThreads(int threads);

    native void torchManualSeed(long seed);

    native void torchShowConfig(Set<String> set);

    native long[] torchSizes(Pointer handle);

    native byte[] torchDataPtr(Pointer handle);

    native int torchDType(Pointer handle);

    native int[] torchDevice(Pointer handle);

    native int torchLayout(Pointer handle);

    native Pointer torchTo(Pointer handle, int dType, int[] device, boolean copy);

    native Pointer torchToSparse(Pointer handle);

    native Pointer torchToDense(Pointer handle);

    native Pointer tensorClone(Pointer handle);

    native Pointer torchEmpty(
            long[] shape, int dType, int layout, int[] device, boolean requiredGrad);

    native Pointer torchZeros(
            long[] shape, int dType, int layout, int[] device, boolean requiredGrad);

    native Pointer torchOnes(
            long[] shape, int dType, int layout, int[] device, boolean requiredGrad);

    native Pointer torchZerosLike(
            Pointer handle, int dType, int layout, int[] device, boolean requiredGrad);

    native Pointer torchOnesLike(
            Pointer handle, int dType, int layout, int[] device, boolean requiredGrad);

    native Pointer torchArange(
            float start,
            float end,
            float step,
            int dType,
            int layout,
            int[] device,
            boolean requiredGrad);

    native Pointer torchLinspace(
            float start,
            float end,
            int step,
            int dType,
            int layout,
            int[] device,
            boolean requiredGrad);

    native Pointer torchAdd(Pointer self, Pointer other);

    native void torchAddi(Pointer self, Pointer other);

    native Pointer torchExpand(Pointer self, long[] shape);

    native Pointer torchSub(Pointer self, Pointer other);

    native void torchSubi(Pointer self, Pointer other);

    native Pointer torchMul(Pointer self, Pointer other);

    native void torchMuli(Pointer self, Pointer other);

    native Pointer torchTrueDivide(Pointer self, Pointer other);

    native void torchTrueDividei(Pointer self, Pointer other);

    native Pointer torchRemainder(Pointer self, Pointer other);

    native void torchRemainderi(Pointer self, Pointer other);

    native Pointer torchPow(Pointer self, Pointer exponent);

    native void torchPowi(Pointer self, Pointer exponent);

    native Pointer torchMatmul(Pointer self, Pointer other);

    native Pointer torchDot(Pointer self, Pointer other);

    native Pointer torchMM(Pointer self, Pointer other);

    native Pointer torchLogicalAnd(Pointer self, Pointer other);

    native Pointer torchLogicalOr(Pointer self, Pointer other);

    native Pointer torchLogicalXor(Pointer self, Pointer other);

    native Pointer torchLogicalNot(Pointer handle);

    native Pointer torchReshape(Pointer handle, long[] shape);

    native Pointer torchSoftmax(Pointer handle, long dim, int dType);

    native Pointer torchLogSoftmax(Pointer handle, long dim, int dType);

    native Pointer torchArgMax(Pointer handle);

    native Pointer torchArgMax(Pointer handle, long dim, boolean keepDim);

    native Pointer torchArgMin(Pointer handle);

    native Pointer torchArgMin(Pointer handle, long dim, boolean keepDim);

    native Pointer torchArgSort(Pointer handle);

    native Pointer torchArgSort(Pointer handle, long dim, boolean keepDim);

    native Pointer torchSort(Pointer handle, long dim, boolean descending);

    native Pointer torchPermute(Pointer handle, long[] dims);

    native Pointer torchTranspose(Pointer handle, long axis1, long axis2);

    native boolean contentEqual(Pointer handle1, Pointer handle2);

    native Pointer torchFromBlob(
            ByteBuffer data,
            long[] shape,
            int dType,
            int layout,
            int[] device,
            boolean requiredGrad);

    native Pointer torchIndex(
            Pointer handle, long[] minIndices, long[] maxIndices, long[] stepIndices);

    native void torchIndexPut(
            Pointer handle,
            Pointer valueHandle,
            long[] minIndices,
            long[] maxIndices,
            long[] stepIndices);

    native void torchSet(Pointer selfHandle, Pointer otherHandle);

    native Pointer torchSlice(Pointer handle, long dim, long start, long end, long step);

    native Pointer torchGather(Pointer handle, Pointer index, long dim, boolean sparseGrad);

    native Pointer torchMaskedSelect(Pointer handle, Pointer maskHandle);

    native void torchMaskedPut(Pointer handle, Pointer valueHandle, Pointer maskHandle);

    native void torchDeleteTensor(Pointer handle);

    native void torchDeleteModule(Pointer handle);

    native void torchDeleteIValue(Pointer handle);

    native Pointer torchMax(Pointer handle);

    native Pointer torchMax(Pointer self, Pointer other);

    native Pointer torchMax(Pointer handle, long dim, boolean keepDim);

    native Pointer torchMin(Pointer handle);

    native Pointer torchMin(Pointer self, Pointer other);

    native Pointer torchMin(Pointer handle, long dim, boolean keepDim);

    native Pointer torchMean(Pointer handle);

    native Pointer torchMean(Pointer handle, long dim, boolean keepDim);

    native Pointer torchSum(Pointer handle);

    native Pointer torchSum(Pointer handle, long[] dim, boolean keepDim);

    native Pointer torchCumSum(Pointer handle, long dim);

    native Pointer torchFlatten(Pointer handle, long startDim, long endDim);

    native Pointer[] torchSplit(Pointer handle, long size, long dim);

    native Pointer[] torchSplit(Pointer handle, long[] indices, long dim);

    native Pointer torchUnsqueeze(Pointer handle, long dim);

    native Pointer torchSqueeze(Pointer handle);

    native Pointer torchSqueeze(Pointer handle, long axis);

    native Pointer torchStack(Pointer[] handles, long dim);

    native Pointer torchCat(Pointer[] handles, long dim);

    native Pointer torchRepeat(Pointer handle, long[] repeats);

    native Pointer torchRepeatInterleave(Pointer handle, long repeat, long axis);

    native Pointer torchAbs(Pointer handle);

    native Pointer torchSquare(Pointer self);

    native Pointer torchFloor(Pointer handle);

    native Pointer torchCeil(Pointer handle);

    native Pointer torchClamp(Pointer handle, Pointer min, Pointer max);

    native Pointer torchRound(Pointer handle);

    native Pointer torchTrunc(Pointer handle);

    native Pointer torchExp(Pointer handle);

    native Pointer torchLog(Pointer handle);

    native Pointer torchLog10(Pointer handle);

    native Pointer torchLog2(Pointer handle);

    native Pointer torchSin(Pointer handle);

    native Pointer torchCos(Pointer handle);

    native Pointer torchTan(Pointer handle);

    native Pointer torchASin(Pointer handle);

    native Pointer torchAcos(Pointer handle);

    native Pointer torchAtan(Pointer handle);

    native Pointer torchSqrt(Pointer handle);

    native Pointer torchSinh(Pointer handle);

    native Pointer torchCosh(Pointer handle);

    native Pointer torchTanh(Pointer handle);

    native Pointer torchSigmoid(Pointer handle);

    native Pointer torchWhere(Pointer handle, Pointer x, Pointer y);

    native Pointer torchAll(Pointer self);

    native Pointer torchAny(Pointer self);

    native Pointer torchNone(Pointer self);

    native Pointer torchEq(Pointer self, Pointer other);

    native Pointer torchNeq(Pointer self, Pointer other);

    native Pointer torchGt(Pointer self, Pointer other);

    native Pointer torchGte(Pointer self, Pointer other);

    native Pointer torchLt(Pointer self, Pointer other);

    native Pointer torchLte(Pointer self, Pointer other);

    native Pointer torchNeg(Pointer self);

    native void torchNegi(Pointer self);

    native Pointer torchIsNaN(Pointer self);

    native Pointer torchIsInf(Pointer self);

    native Pointer atNormal(
            double mean,
            double std,
            long[] sizes,
            int dType,
            int layout,
            int[] device,
            boolean requiredGrad);

    native Pointer tensorUniform(
            double from,
            double to,
            long[] sizes,
            int dType,
            int layout,
            int[] device,
            boolean requiredGrad);

    native Pointer torchEye(
            int n, int m, int dType, int layout, int[] device, boolean requiredGrad);

    native Pointer torchUpsampleBilinear2d(Pointer handle, long[] shape, boolean alignCorners);

    native Pointer torchNNLinear(
            Pointer handle, Pointer weightHandle, Pointer biasHandle, boolean bias);

    native Pointer torchNNRelu(Pointer handle);

    native Pointer torchNNConvNd(
            int dim,
            Pointer inputHandle,
            Pointer weightHandle,
            Pointer biasHandle,
            long[] stride,
            long[] pad,
            long[] dilation,
            int numGroup,
            boolean bias);

    native Pointer torchNNDropout(Pointer inputHandle, double probability, boolean isTrain);

    native Pointer torchNNBatchNorm(
            Pointer inputHandle,
            Pointer weigthHandle,
            Pointer biasHandle,
            Pointer runningMeanHandle,
            Pointer runningVarHandle,
            boolean isTraining,
            double momentum,
            double eps);

    native Pointer torchNNAvgPool(
            Pointer inputHandle,
            int dim,
            long[] kernel,
            long[] stride,
            long[] pad,
            boolean useCeil,
            boolean countIncludePad);

    native Pointer torchNNMaxPool(
            Pointer inputHandle,
            int dim,
            long[] kernel,
            long[] stride,
            long[] pad,
            boolean useCeil);

    native Pointer torchNNAdaptiveAvgPool(Pointer inputHandle, int dim, long[] outSize);

    native Pointer torchNNAdaptiveMaxPool(Pointer inputHandle, int dim, long[] outSize);

    native boolean torchRequiresGrad(Pointer inputHandle);

    native String torchGradFnName(Pointer inputHandle);

    native void torchAttachGrad(Pointer inputHandle);

    native Pointer torchGrad(Pointer inputHandle);

    native Pointer torchDetachGrad(Pointer inputHandle);

    native void torchBackward(
            Pointer inputHandle, Pointer gradHandle, boolean keepGraph, boolean createGraph);

    native Pointer moduleLoad(String path, int[] device);

    native void moduleEval(Pointer handle);

    native void moduleTrain(Pointer handle);

    native Pointer moduleForward(Pointer moduleHandle, Pointer[] arrayHandles, boolean isTrain);

    native Pointer iValueCreateFromTensor(Pointer tensorHandle);

    native Pointer iValueToTensor(Pointer iValueHandle);

    native Pointer[] iValueToTensorList(Pointer iValueHandle);

    native Pointer[] iValueToList(Pointer iValueHandle);

    native Pointer[] iValueToListFromTuple(Pointer iValueHandle);

    native Pointer[] iValueToMap(Pointer iValueHandle);

    native String iValueToString(Pointer iValueHandle);

    native boolean iValueIsString(Pointer iValueHandle);

    native boolean iValueIsTensor(Pointer iValueHandle);

    native boolean iValueIsTensorList(Pointer iValueHandle);

    native boolean iValueIsList(Pointer iValueHandle);

    native boolean iValueIsMap(Pointer iValueHandle);

    native boolean iValueIsTuple(Pointer iValueHandle);

    native void adamUpdate(
            Pointer weight,
            Pointer grad,
            Pointer mean,
            Pointer variance,
            float lr,
            float wd,
            float rescaleGrad,
            float clipGrad,
            float beta1,
            float beta2,
            float eps);

    native void sgdUpdate(
            Pointer weight,
            Pointer grad,
            Pointer state,
            float lr,
            float wd,
            float rescaleGrad,
            float clipGrad,
            float momentum);
}
