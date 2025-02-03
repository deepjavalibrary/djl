/*
 * Copyright 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.engine.rust;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.NDScope;
import ai.djl.ndarray.NDUtils;
import ai.djl.ndarray.index.NDArrayIndexer;
import ai.djl.ndarray.internal.NDArrayEx;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.ndarray.types.SparseFormat;
import ai.djl.nn.recurrent.RNN;

import java.util.List;

/** {@code PtNDArrayEx} is the Rust implementation of the {@link NDArrayEx}. */
@SuppressWarnings("try")
public class RsNDArrayEx implements NDArrayEx {

    private RsNDArray array;

    /**
     * Constructs an {@code PtNDArrayEx} given a {@link NDArray}.
     *
     * @param parent the {@link NDArray} to extend
     */
    RsNDArrayEx(RsNDArray parent) {
        this.array = parent;
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray rdivi(NDArray b) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray rmodi(NDArray b) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray rpowi(Number n) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray relu() {
        return new RsNDArray(array.getManager(), RustLibrary.relu(array.getHandle()));
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray sigmoid() {
        return new RsNDArray(array.getManager(), RustLibrary.sigmoid(array.getHandle()));
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray tanh() {
        return array.tanh();
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray softPlus() {
        return new RsNDArray(array.getManager(), RustLibrary.softPlus(array.getHandle()));
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray softSign() {
        return new RsNDArray(array.getManager(), RustLibrary.softSign(array.getHandle()));
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray leakyRelu(float alpha) {
        return new RsNDArray(array.getManager(), RustLibrary.leakyRelu(array.getHandle(), alpha));
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray elu(float alpha) {
        return new RsNDArray(array.getManager(), RustLibrary.elu(array.getHandle(), alpha));
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray selu() {
        return new RsNDArray(array.getManager(), RustLibrary.selu(array.getHandle()));
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray gelu() {
        return new RsNDArray(array.getManager(), RustLibrary.gelu(array.getHandle()));
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray maxPool(Shape kernelShape, Shape stride, Shape padding, boolean ceilMode) {
        return new RsNDArray(
                array.getManager(),
                RustLibrary.maxPool(
                        array.getHandle(),
                        kernelShape.getShape(),
                        stride.getShape(),
                        padding.getShape(),
                        ceilMode));
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray globalMaxPool() {
        Shape shape = getPoolShape(array);
        long newHandle = RustLibrary.adaptiveMaxPool(array.getHandle(), shape.getShape());
        try (NDArray temp = new RsNDArray(array.getManager(), newHandle)) {
            return (RsNDArray) temp.reshape(array.getShape().slice(0, 2));
        }
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray avgPool(
            Shape kernelShape,
            Shape stride,
            Shape padding,
            boolean ceilMode,
            boolean countIncludePad) {
        if (kernelShape.size() != 2) {
            throw new UnsupportedOperationException("Only avgPool2d is supported");
        }
        return new RsNDArray(
                array.getManager(),
                RustLibrary.avgPool2d(
                        array.getHandle(), kernelShape.getShape(), stride.getShape()));
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray globalAvgPool() {
        Shape shape = getPoolShape(array);
        long newHandle = RustLibrary.adaptiveAvgPool(array.getHandle(), shape.getShape());
        try (NDArray temp = new RsNDArray(array.getManager(), newHandle)) {
            return (RsNDArray) temp.reshape(array.getShape().slice(0, 2));
        }
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray lpPool(
            float normType, Shape kernelShape, Shape stride, Shape padding, boolean ceilMode) {
        if (padding.size() != 0) {
            throw new IllegalArgumentException("padding is not supported for Rust engine");
        }
        return new RsNDArray(
                array.getManager(),
                RustLibrary.lpPool(
                        array.getHandle(),
                        normType,
                        kernelShape.getShape(),
                        stride.getShape(),
                        ceilMode));
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray globalLpPool(float normType) {
        long[] kernelShape = array.getShape().slice(2).getShape();
        long[] stride = getPoolShape(array).getShape();
        long newHandle =
                RustLibrary.lpPool(array.getHandle(), normType, kernelShape, stride, false);
        try (NDArray temp = new RsNDArray(array.getManager(), newHandle)) {
            return (RsNDArray) temp.reshape(array.getShape().slice(0, 2));
        }
    }

    /** {@inheritDoc} */
    @Override
    public void adadeltaUpdate(
            NDList inputs,
            NDList weights,
            float weightDecay,
            float rescaleGrad,
            float clipGrad,
            float rho,
            float epsilon) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public void adagradUpdate(
            NDList inputs,
            NDList weights,
            float learningRate,
            float weightDecay,
            float rescaleGrad,
            float clipGrad,
            float epsilon) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public void adamUpdate(
            NDList inputs,
            NDList weights,
            float learningRate,
            float learningRateBiasCorrection,
            float weightDecay,
            float rescaleGrad,
            float clipGrad,
            float beta1,
            float beta2,
            float epsilon,
            boolean lazyUpdate,
            boolean adamw) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public void nagUpdate(
            NDList inputs,
            NDList weights,
            float learningRate,
            float weightDecay,
            float rescaleGrad,
            float clipGrad,
            float momentum) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public void rmspropUpdate(
            NDList inputs,
            NDList weights,
            float learningRate,
            float weightDecay,
            float rescaleGrad,
            float clipGrad,
            float rho,
            float momentum,
            float epsilon,
            boolean centered) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public void sgdUpdate(
            NDList inputs,
            NDList weights,
            float learningRate,
            float weightDecay,
            float rescaleGrad,
            float clipGrad,
            float momentum,
            boolean lazyUpdate) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDList convolution(
            NDArray input,
            NDArray weight,
            NDArray bias,
            Shape stride,
            Shape padding,
            Shape dilation,
            int groups) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDList deconvolution(
            NDArray input,
            NDArray weight,
            NDArray bias,
            Shape stride,
            Shape padding,
            Shape outPadding,
            Shape dilation,
            int groups) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDList linear(NDArray input, NDArray weight, NDArray bias) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDList embedding(NDArray input, NDArray weight, SparseFormat sparseFormat) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDList prelu(NDArray input, NDArray alpha) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDList dropout(NDArray input, float rate, boolean training) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDList layerNorm(
            NDArray input, Shape normalizedShape, NDArray gamma, NDArray beta, float eps) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDList batchNorm(
            NDArray input,
            NDArray runningMean,
            NDArray runningVar,
            NDArray gamma,
            NDArray beta,
            int axis,
            float momentum,
            float eps,
            boolean training) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDList rnn(
            NDArray input,
            NDArray state,
            NDList params,
            boolean hasBiases,
            int numLayers,
            RNN.Activation activation,
            double dropRate,
            boolean training,
            boolean bidirectional,
            boolean batchFirst) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDList gru(
            NDArray input,
            NDArray state,
            NDList params,
            boolean hasBiases,
            int numLayers,
            double dropRate,
            boolean training,
            boolean bidirectional,
            boolean batchFirst) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDList lstm(
            NDArray input,
            NDList states,
            NDList params,
            boolean hasBiases,
            int numLayers,
            double dropRate,
            boolean training,
            boolean bidirectional,
            boolean batchFirst) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray interpolation(long[] size, int mode, boolean alignCorners) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray resize(int width, int height, int interpolation) {
        long[] shape = array.getShape().getShape();
        if (shape[0] == height && shape[1] == width) {
            return array.toType(DataType.FLOAT32, false);
        }
        // TODO:
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray randomFlipLeftRight() {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray randomFlipTopBottom() {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray randomBrightness(float brightness) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray randomHue(float hue) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray randomColorJitter(
            float brightness, float contrast, float saturation, float hue) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArrayIndexer getIndexer(NDManager manager) {
        return new RsNDArrayIndexer((RsNDManager) manager);
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray where(NDArray condition, NDArray other) {
        // Try to broadcast if shape mismatch
        if (!condition.getShape().equals(array.getShape())) {
            throw new UnsupportedOperationException(
                    "condition and self shape mismatch, broadcast is not supported");
        }
        RsNDManager manager = array.getManager();
        try (NDScope ignore = new NDScope()) {
            long conditionHandle = manager.from(condition).getHandle();
            long otherHandle = manager.from(other).getHandle();
            RsNDArray ret =
                    new RsNDArray(
                            manager,
                            RustLibrary.where(conditionHandle, array.getHandle(), otherHandle));
            NDScope.unregister(ret);
            return ret;
        }
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray stack(NDList arrays, int axis) {
        long[] srcArray = new long[arrays.size() + 1];
        srcArray[0] = array.getHandle();
        RsNDManager manager = array.getManager();

        try (NDScope ignore = new NDScope()) {
            int i = 1;
            for (NDArray arr : arrays) {
                srcArray[i++] = manager.from(arr).getHandle();
            }
            RsNDArray ret = new RsNDArray(manager, RustLibrary.stack(srcArray, axis));
            NDScope.unregister(ret);
            return ret;
        }
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray concat(NDList list, int axis) {
        NDUtils.checkConcatInput(list);

        long[] srcArray = new long[list.size() + 1];
        srcArray[0] = array.getHandle();
        RsNDManager manager = array.getManager();
        try (NDScope ignore = new NDScope()) {
            int i = 1;
            for (NDArray arr : list) {
                srcArray[i++] = manager.from(arr).getHandle();
            }
            RsNDArray ret = new RsNDArray(manager, RustLibrary.concat(srcArray, axis));
            NDScope.unregister(ret);
            return ret;
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDList multiBoxTarget(
            NDList inputs,
            float iouThreshold,
            float ignoreLabel,
            float negativeMiningRatio,
            float negativeMiningThreshold,
            int minNegativeSamples) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDList multiBoxPrior(
            List<Float> sizes,
            List<Float> ratios,
            List<Float> steps,
            List<Float> offsets,
            boolean clip) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDList multiBoxDetection(
            NDList inputs,
            boolean clip,
            float threshold,
            int backgroundId,
            float nmsThreshold,
            boolean forceSuppress,
            int nmsTopK) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public RsNDArray getArray() {
        return array;
    }

    private Shape getPoolShape(NDArray array) {
        switch (array.getShape().dimension() - 2) {
            case 1:
                return new Shape(1);
            case 2:
                return new Shape(1, 1);
            case 3:
                return new Shape(1, 1, 1);
            default:
                throw new IllegalArgumentException("the input dimension should be in [3, 5]");
        }
    }
}
