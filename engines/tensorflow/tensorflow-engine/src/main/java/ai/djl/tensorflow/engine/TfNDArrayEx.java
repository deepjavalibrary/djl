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

package ai.djl.tensorflow.engine;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDUtils;
import ai.djl.ndarray.index.NDArrayIndexer;
import ai.djl.ndarray.internal.NDArrayEx;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.ndarray.types.SparseFormat;
import ai.djl.nn.recurrent.RNN;
import java.util.List;
import org.tensorflow.internal.c_api.TFE_TensorHandle;

public class TfNDArrayEx implements NDArrayEx {

    private TfNDArray array;

    /**
     * Constructs an {@code MxNDArrayEx} given a {@link NDArray}.
     *
     * @param array the array
     */
    TfNDArrayEx(TfNDArray array) {
        this.array = array;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rdiv(Number n) {
        return rdiv(array.getManager().create(n).toType(array.getDataType(), false));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rdiv(NDArray b) {
        return b.div(array);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rdivi(Number n) {
        return rdivi(array.getManager().create(n).toType(array.getDataType(), false));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rdivi(NDArray b) {
        TFE_TensorHandle newHandle =
                array.getManager()
                        .opExecutor("Div")
                        .addInput(b)
                        .addInput(array)
                        .buildRawPointer(1)[0];
        array.setHandle(newHandle);
        return array;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rsub(Number n) {
        return rsub(array.getManager().create(n).toType(array.getDataType(), false));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rsub(NDArray b) {
        return b.sub(array);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rsubi(Number n) {
        return rsubi(array.getManager().create(n).toType(array.getDataType(), false));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rsubi(NDArray b) {
        TFE_TensorHandle newHandle =
                array.getManager()
                        .opExecutor("Sub")
                        .addInput(b)
                        .addInput(array)
                        .buildRawPointer(1)[0];
        array.setHandle(newHandle);
        return array;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rmod(Number n) {
        return rmod(array.getManager().create(n).toType(array.getDataType(), false));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rmod(NDArray b) {
        return b.mod(array);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rmodi(Number n) {
        return rmodi(array.getManager().create(n).toType(array.getDataType(), false));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rmodi(NDArray b) {
        TFE_TensorHandle newHandle =
                array.getManager()
                        .opExecutor("Mod")
                        .addInput(b)
                        .addInput(array)
                        .buildRawPointer(1)[0];
        array.setHandle(newHandle);
        return array;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rpow(Number n) {
        return array.getManager().create(n).toType(array.getDataType(), false).pow(array);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rpowi(Number n) {
        TfNDManager manager = array.getManager();
        try (NDArray temp = manager.create(n);
                NDArray casted = temp.toType(array.getDataType(), false)) {
            TFE_TensorHandle newHandle =
                    manager.opExecutor("Pow")
                            .addInput(casted)
                            .addInput(array)
                            .buildRawPointer(1)[0];
            array.setHandle(newHandle);
            return array;
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray relu() {
        return array.getManager().opExecutor("Relu").addInput(array).buildSingletonOrThrow();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sigmoid() {
        return array.getManager().opExecutor("Sigmoid").addInput(array).buildSingletonOrThrow();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray tanh() {
        return array.tanh();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray softPlus() {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    @SuppressWarnings({"rawtypes", "unchecked"})
    public NDArray softSign() {
        return array.getManager().opExecutor("Softsign").addInput(array).buildSingletonOrThrow();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray leakyRelu(float alpha) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray elu(float alpha) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    @SuppressWarnings({"rawtypes", "unchecked"})
    public NDArray selu() {
        return array.getManager().opExecutor("Selu").addInput(array).buildSingletonOrThrow();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray gelu() {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray maxPool(Shape kernelShape, Shape stride, Shape padding, boolean ceilMode) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray globalMaxPool() {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray avgPool(
            Shape kernelShape,
            Shape stride,
            Shape padding,
            boolean ceilMode,
            boolean countIncludePad) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray globalAvgPool() {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray lpPool(
            float normType, Shape kernelShape, Shape stride, Shape padding, boolean ceilMode) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray globalLpPool(float normType) {
        throw new UnsupportedOperationException("Not implemented");
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
            float weightDecay,
            float rescaleGrad,
            float clipGrad,
            float beta1,
            float beta2,
            float epsilon,
            boolean lazyUpdate) {
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
    public NDList embedding(NDArray input, NDArray weight, SparseFormat sparse) {
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
        throw new UnsupportedOperationException();
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
            boolean train,
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
    public NDArray normalize(float[] mean, float[] std) {
        // TODO: TensorFlow does not support channels first on CPU for conv2d
        // https://github.com/tensorflow/tensorflow/issues/32691
        // https://github.com/tensorflow/tensorflow/issues/26411
        int dim = getArray().getShape().dimension();
        Shape shape = (dim == 3) ? new Shape(1, 1, 3) : new Shape(1, 1, 1, 3);
        TfNDManager manager = array.getManager();
        try (NDArray meanArr = manager.create(mean, shape);
                NDArray stdArr = manager.create(std, shape)) {
            return getArray().sub(meanArr).divi(stdArr);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray toTensor() {
        // TODO: TensorFlow does not support channels first on CPU for conv2d
        // https://github.com/tensorflow/tensorflow/issues/32691
        // https://github.com/tensorflow/tensorflow/issues/26411
        TfNDManager manager = array.getManager();
        try (TfNDManager subManager = (TfNDManager) manager.newSubManager()) {
            array.attach(subManager);
            NDArray input = array;
            int dim = input.getShape().dimension();
            if (dim == 3) {
                input = input.expandDims(0);
            }
            input = input.div(255.0);
            if (dim == 3) {
                input = input.squeeze(0);
            }
            // The network by default takes float32
            NDArray output =
                    (!input.getDataType().equals(DataType.FLOAT32))
                            ? input.toType(DataType.FLOAT32, false)
                            : input;
            array.attach(manager);
            output.attach(manager);
            return output;
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray resize(int width, int height, int interpolation) {
        if (array.isEmpty()) {
            throw new IllegalArgumentException("Can't resize image with 0 dims.");
        }
        String op = getResizeOpName(interpolation);
        TfNDManager manager = array.getManager();
        if (array.getShape().dimension() == 3) {
            try (NDArray temp = array.expandDims(0);
                    NDArray size = manager.create(new int[] {height, width});
                    NDArray image =
                            manager.opExecutor(op)
                                    .addInput(temp)
                                    .addInput(size)
                                    .buildSingletonOrThrow()) {
                return image.squeeze();
            }
        }
        try (NDArray size = manager.create(new int[] {height, width})) {
            return manager.opExecutor(op).addInput(array).addInput(size).buildSingletonOrThrow();
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
        return new TfNDArrayIndexer(array.getManager());
    }

    /** {@inheritDoc} */
    @Override
    public NDArray where(NDArray condition, NDArray other) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray stack(NDList arrays) {
        return stack(arrays, 0);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray stack(NDList arrays, int axis) {
        NDArray[] srcArray = new NDArray[arrays.size() + 1];
        srcArray[0] = array;
        System.arraycopy(arrays.toArray(new NDArray[0]), 0, srcArray, 1, arrays.size());
        return array.getManager()
                .opExecutor("Pack")
                .addInputList(srcArray)
                .addParam("axis", axis)
                .buildSingletonOrThrow();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray concat(NDList arrays, int axis) {
        NDUtils.checkConcatInput(arrays);
        NDArray[] srcArray = new NDArray[arrays.size() + 1];
        srcArray[0] = array;
        System.arraycopy(arrays.toArray(new NDArray[0]), 0, srcArray, 1, arrays.size());
        TfNDManager manager = array.getManager();
        try (NDArray axisArr = manager.create(axis)) {
            return manager.opExecutor("ConcatV2")
                    .addInputList(srcArray)
                    .addInput(axisArr)
                    .buildSingletonOrThrow();
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
    public NDArray getArray() {
        return array;
    }

    private String getResizeOpName(int interpolate) {
        switch (interpolate) {
            case 0:
                return "ResizeNearestNeighbor";
            case 1:
                return "ResizeBilinear";
            case 2:
                return "ResizeArea";
            case 3:
                return "ResizeBicubic";
            default:
                throw new UnsupportedOperationException(
                        "The kind of interpolation is not supported.");
        }
    }
}
