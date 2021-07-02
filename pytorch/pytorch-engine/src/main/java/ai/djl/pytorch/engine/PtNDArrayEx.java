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
package ai.djl.pytorch.engine;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.NDUtils;
import ai.djl.ndarray.index.NDArrayIndexer;
import ai.djl.ndarray.internal.NDArrayEx;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.ndarray.types.SparseFormat;
import ai.djl.nn.recurrent.RNN;
import ai.djl.pytorch.jni.JniUtils;
import java.util.List;

/** {@code PtNDArrayEx} is the PyTorch implementation of the {@link NDArrayEx}. */
public class PtNDArrayEx implements NDArrayEx {

    private static final NDArrayIndexer INDEXER = new PtNDArrayIndexer();

    private PtNDArray array;

    /**
     * Constructs an {@code PtNDArrayEx} given a {@link NDArray}.
     *
     * @param parent the {@link NDArray} to extend
     */
    PtNDArrayEx(PtNDArray parent) {
        this.array = parent;
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray rdiv(Number n) {
        return rdiv(array.getManager().create(n));
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray rdiv(NDArray b) {
        return (PtNDArray) b.div(array);
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray rdivi(Number n) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray rdivi(NDArray b) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray rsub(Number n) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray rsub(NDArray b) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray rsubi(Number n) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray rsubi(NDArray b) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray rmod(Number n) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray rmod(NDArray b) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray rmodi(Number n) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray rmodi(NDArray b) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray rpow(Number n) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray rpowi(Number n) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray relu() {
        return JniUtils.relu(array);
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray sigmoid() {
        return JniUtils.sigmoid(array);
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray tanh() {
        return JniUtils.tanh(array);
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray softPlus() {
        return JniUtils.softPlus(array);
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray softSign() {
        return JniUtils.softSign(array);
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray leakyRelu(float alpha) {
        return JniUtils.leakyRelu(array, alpha);
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray elu(float alpha) {
        return JniUtils.elu(array, alpha);
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray selu() {
        return JniUtils.selu(array);
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray gelu() {
        return JniUtils.gelu(array);
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray maxPool(Shape kernelShape, Shape stride, Shape padding, boolean ceilMode) {
        return JniUtils.maxPool(array, kernelShape, stride, padding, ceilMode);
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray globalMaxPool() {
        Shape shape = getPoolShape(array);
        try (NDArray temp = JniUtils.adaptiveMaxPool(array, shape)) {
            return (PtNDArray) temp.reshape(array.getShape().slice(0, 2));
        }
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray avgPool(
            Shape kernelShape,
            Shape stride,
            Shape padding,
            boolean ceilMode,
            boolean countIncludePad) {
        return JniUtils.avgPool(array, kernelShape, stride, padding, ceilMode, countIncludePad);
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray globalAvgPool() {
        Shape shape = getPoolShape(array);
        try (NDArray temp = JniUtils.adaptiveAvgPool(array, shape)) {
            return (PtNDArray) temp.reshape(array.getShape().slice(0, 2));
        }
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray lpPool(
            float normType, Shape kernelShape, Shape stride, Shape padding, boolean ceilMode) {
        if (padding.size() != 0) {
            throw new IllegalArgumentException("padding is not supported for PyTorch engine");
        }
        return JniUtils.lpPool(array, normType, kernelShape, stride, ceilMode);
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray globalLpPool(float normType) {
        try (NDArray temp =
                JniUtils.lpPool(
                        array, normType, array.getShape().slice(2), getPoolShape(array), false)) {
            return (PtNDArray) temp.reshape(array.getShape().slice(0, 2));
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
        throw new UnsupportedOperationException(
                "AdaDelta optimzier is not supported for PyTorch engine!");
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
            float weightDecay,
            float rescaleGrad,
            float clipGrad,
            float beta1,
            float beta2,
            float epsilon,
            boolean lazyUpdate) {
        // TODO: Lazy update not used
        JniUtils.adamUpdate(
                (PtNDArray) inputs.get(0),
                (PtNDArray) inputs.get(1),
                (PtNDArray) inputs.get(2),
                (PtNDArray) inputs.get(3),
                learningRate,
                weightDecay,
                rescaleGrad,
                clipGrad,
                beta1,
                beta2,
                epsilon);
        // call zero-grad
        JniUtils.zeroGrad((PtNDArray) weights.singletonOrThrow());
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
        // TODO: Lazy update not used
        JniUtils.sgdUpdate(
                (PtNDArray) inputs.get(0),
                (PtNDArray) inputs.get(1),
                (momentum == 0f) ? null : (PtNDArray) inputs.get(2),
                learningRate,
                weightDecay,
                rescaleGrad,
                clipGrad,
                momentum);
        // call zero-grad
        JniUtils.zeroGrad((PtNDArray) weights.singletonOrThrow());
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
        return new NDList(
                JniUtils.convolution(
                        (PtNDArray) input,
                        (PtNDArray) weight,
                        (PtNDArray) bias,
                        stride,
                        padding,
                        dilation,
                        groups));
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
        return new NDList(JniUtils.linear((PtNDArray) input, (PtNDArray) weight, (PtNDArray) bias));
    }

    /** {@inheritDoc} */
    @Override
    public NDList embedding(NDArray input, NDArray weight, SparseFormat sparseFormat) {
        if (!sparseFormat.equals(SparseFormat.DENSE) && !sparseFormat.equals(SparseFormat.COO)) {
            throw new IllegalArgumentException("PyTorch only supports COO");
        }
        return new NDList(
                JniUtils.embedding(
                        (PtNDArray) input,
                        (PtNDArray) weight,
                        sparseFormat.equals(SparseFormat.COO)));
    }

    /** {@inheritDoc} */
    @Override
    public NDList prelu(NDArray input, NDArray alpha) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDList dropout(NDArray input, float rate, boolean training) {
        return new NDList(JniUtils.dropout((PtNDArray) input, rate, training));
    }

    /** {@inheritDoc} */
    @Override
    public NDList layerNorm(
            NDArray input, Shape normalizedShape, NDArray gamma, NDArray beta, float eps) {
        return new NDList(
                JniUtils.layerNorm(
                        (PtNDArray) input,
                        normalizedShape,
                        (PtNDArray) gamma,
                        (PtNDArray) beta,
                        eps));
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
        // TODO PyTorch will support axis argument
        // https://github.com/pytorch/pytorch/issues/21856
        if (axis == -1) {
            return new NDList(
                    JniUtils.batchNorm(
                            (PtNDArray) input,
                            (PtNDArray) runningMean,
                            (PtNDArray) runningVar,
                            (PtNDArray) gamma,
                            (PtNDArray) beta,
                            training,
                            // momentum is defined differently in PyTorch
                            1f - momentum,
                            eps));
        }
        // apply the swapAxes to simulate BatchNorm with axis
        try (NDManager subManager = input.getManager().newSubManager()) {
            input.attach(subManager);
            NDArray result = input;
            result = result.swapAxes(1, axis);
            result =
                    JniUtils.batchNorm(
                            (PtNDArray) result,
                            (PtNDArray) runningMean,
                            (PtNDArray) runningVar,
                            (PtNDArray) gamma,
                            (PtNDArray) beta,
                            training,
                            // momentum is defined differently in PyTorch
                            1f - momentum,
                            eps);
            result = result.swapAxes(1, axis);
            input.attach(subManager.getParentManager());
            result.attach(subManager.getParentManager());
            return new NDList(result);
        }
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
        return JniUtils.rnn(
                (PtNDArray) input,
                (PtNDArray) state,
                params,
                hasBiases,
                numLayers,
                activation,
                dropRate,
                training,
                bidirectional,
                batchFirst);
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
        return JniUtils.gru(
                (PtNDArray) input,
                (PtNDArray) state,
                params,
                hasBiases,
                numLayers,
                dropRate,
                training,
                bidirectional,
                batchFirst);
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
        return JniUtils.lstm(
                (PtNDArray) input,
                states,
                params,
                hasBiases,
                numLayers,
                dropRate,
                training,
                bidirectional,
                batchFirst);
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray resize(int width, int height, int interpolation) {
        // create subManager to help close intermediate NDArray
        try (NDManager subManager = array.getManager().newSubManager()) {
            array.attach(subManager);
            NDArray result = array;
            if (result.isEmpty()) {
                throw new IllegalArgumentException("attempt to resize of an empty NDArray");
            }
            if (result.getDataType() != DataType.FLOAT32) {
                result = result.toType(DataType.FLOAT32, true);
            }
            int dim = result.getShape().dimension();
            if (dim == 3) {
                result = result.expandDims(0);
            }
            result = result.transpose(0, 3, 1, 2);
            result =
                    JniUtils.interpolate(
                                    (PtNDArray) result,
                                    new long[] {height, width},
                                    getInterpolationMode(interpolation),
                                    false)
                            .transpose(0, 2, 3, 1);
            if (dim == 3) {
                result = result.squeeze(0);
            }
            array.attach(subManager.getParentManager());
            result.attach(subManager.getParentManager());
            return (PtNDArray) result;
        }
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
    public NDArrayIndexer getIndexer() {
        return INDEXER;
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray where(NDArray condition, NDArray other) {
        // Try to broadcast if shape mismatch
        if (!condition.getShape().equals(array.getShape())) {
            throw new UnsupportedOperationException(
                    "condition and self shape mismatch, broadcast is not supported");
        }
        return JniUtils.where((PtNDArray) condition, array, (PtNDArray) other);
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray stack(NDList arrays, int axis) {
        PtNDArray[] srcArray = new PtNDArray[arrays.size() + 1];
        srcArray[0] = array;
        System.arraycopy(arrays.toArray(new NDArray[0]), 0, srcArray, 1, arrays.size());
        return JniUtils.stack(srcArray, axis);
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray concat(NDList list, int axis) {
        NDUtils.checkConcatInput(list);

        PtNDArray[] srcArray = new PtNDArray[list.size() + 1];
        srcArray[0] = array;
        System.arraycopy(list.toArray(new NDArray[0]), 0, srcArray, 1, list.size());
        return JniUtils.cat(srcArray, axis);
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
    public PtNDArray getArray() {
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

    // Here is the list of PyTorch C++ interpolation mapping: kNearest, kLinear, kBilinear,
    // kBicubic, kTrilinear, kArea
    private int getInterpolationMode(int interpolation) {
        switch (interpolation) {
            case 0:
                return 0;
            case 1:
                return 2;
            case 2:
                return 5;
            case 3:
                return 3;
            default:
                throw new UnsupportedOperationException(
                        "The kind of interpolation is not supported.");
        }
    }
}
