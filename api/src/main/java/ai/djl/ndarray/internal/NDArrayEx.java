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
package ai.djl.ndarray.internal;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDArrayIndexer;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.ndarray.types.SparseFormat;
import ai.djl.nn.Activation;
import ai.djl.nn.recurrent.RNN;
import java.util.List;

/** An internal interface that encapsulates engine specific operations. */
@SuppressWarnings("MissingJavadocMethod")
public interface NDArrayEx {

    ////////////////////////////////////////
    // NDArrays
    ////////////////////////////////////////

    /**
     * Applies reverse division with a scalar - i.e., (n / thisArrayValues).
     *
     * @param n the Value to use for reverse division
     * @return a copy of the array after applying reverse division
     */
    NDArray rdiv(Number n);

    /**
     * Applies reverse division with a scalar - i.e., (n / thisArrayValues).
     *
     * @param b the ndarray to use for reverse division
     * @return a copy of the array after applying reverse division
     */
    NDArray rdiv(NDArray b);

    /**
     * Applies in place reverse division - i.e., (n / thisArrayValues).
     *
     * @param n the value to use for reverse division
     * @return this array after applying reverse division
     */
    NDArray rdivi(Number n);

    /**
     * Applies in place reverse division - i.e., (n / thisArrayValues).
     *
     * @param b the ndarray to use for reverse division
     * @return this array after applying reverse division
     */
    NDArray rdivi(NDArray b);

    /**
     * Applies reverse subtraction with duplicates - i.e., (n - thisArrayValues).
     *
     * @param n the value to use for reverse subtraction
     * @return a copy of array after reverse subtraction
     */
    NDArray rsub(Number n);

    /**
     * Applies reverse subtraction with duplicates - i.e., (n - thisArrayValues).
     *
     * @param b the ndarray to use for reverse subtraction
     * @return a copy of the array after reverse subtraction
     */
    NDArray rsub(NDArray b);

    /**
     * Applies reverse subtraction in place - i.e., (n - thisArrayValues).
     *
     * @param n the value to use for reverse subtraction
     * @return this array after reverse subtraction
     */
    NDArray rsubi(Number n);

    /**
     * Applies reverse subtraction in place - i.e., (n - thisArrayValues).
     *
     * @param b the ndarray to use for reverse subtraction
     * @return this array after reverse subtraction
     */
    NDArray rsubi(NDArray b);

    /**
     * Applies reverse remainder of division with a scalar.
     *
     * @param n the value to use for reverse division
     * @return a copy of array after applying reverse division
     */
    NDArray rmod(Number n);

    /**
     * Applies reverse remainder of division.
     *
     * @param b the ndarray to use for reverse division
     * @return a copy of array after applying reverse division
     */
    NDArray rmod(NDArray b);

    /**
     * Applies in place reverse remainder of division with a scalar.
     *
     * @param n the value to use for reverse division
     * @return this array after applying reverse division
     */
    NDArray rmodi(Number n);

    /**
     * Applies in place reverse remainder of division.
     *
     * @param b the ndarray to use for reverse division
     * @return this array after applying reverse division
     */
    NDArray rmodi(NDArray b);

    /**
     * Reverses the power of each element being raised in the {@code NDArray}.
     *
     * @param n the value to use for reverse power
     * @return a copy of array after applying reverse power
     */
    NDArray rpow(Number n);

    /**
     * Reverses the power of each element being raised in the {@code NDArray} in place.
     *
     * @param n the value to use for reverse power
     * @return a copy of array after applying reverse power
     */
    NDArray rpowi(Number n);

    ////////////////////////////////////////
    // Activations
    ////////////////////////////////////////

    /**
     * Computes rectified linear activation.
     *
     * @return a copy of array after applying relu
     */
    NDArray relu();

    NDArray sigmoid();

    NDArray tanh();

    NDArray softPlus();

    NDArray softSign();

    NDArray leakyRelu(float alpha);

    NDArray elu(float alpha);

    NDArray selu();

    NDArray gelu();

    default NDArray swish(float beta) {
        return Activation.sigmoid(getArray().mul(beta)).mul(getArray());
    }

    default NDArray mish() {
        return getArray().exp().add(1).log2().tanh().mul(getArray());
    }

    ////////////////////////////////////////
    // Pooling Operations
    ////////////////////////////////////////

    NDArray maxPool(Shape kernelShape, Shape stride, Shape padding, boolean ceilMode);

    NDArray globalMaxPool();

    NDArray avgPool(
            Shape kernelShape,
            Shape stride,
            Shape padding,
            boolean ceilMode,
            boolean countIncludePad);

    NDArray globalAvgPool();

    NDArray lpPool(
            float normType, Shape kernelShape, Shape stride, Shape padding, boolean ceilMode);

    NDArray globalLpPool(float normType);

    ////////////////////////////////////////
    // Optimizer
    ////////////////////////////////////////

    void adadeltaUpdate(
            NDList inputs,
            NDList weights,
            float weightDecay,
            float rescaleGrad,
            float clipGrad,
            float rho,
            float epsilon);

    void adagradUpdate(
            NDList inputs,
            NDList weights,
            float learningRate,
            float weightDecay,
            float rescaleGrad,
            float clipGrad,
            float epsilon);

    void adamUpdate(
            NDList inputs,
            NDList weights,
            float learningRate,
            float weightDecay,
            float rescaleGrad,
            float clipGrad,
            float beta1,
            float beta2,
            float epsilon,
            boolean lazyUpdate);

    void nagUpdate(
            NDList inputs,
            NDList weights,
            float learningRate,
            float weightDecay,
            float rescaleGrad,
            float clipGrad,
            float momentum);

    void rmspropUpdate(
            NDList inputs,
            NDList weights,
            float learningRate,
            float weightDecay,
            float rescaleGrad,
            float clipGrad,
            float rho,
            float momentum,
            float epsilon,
            boolean centered);

    void sgdUpdate(
            NDList inputs,
            NDList weights,
            float learningRate,
            float weightDecay,
            float rescaleGrad,
            float clipGrad,
            float momentum,
            boolean lazyUpdate);

    ////////////////////////////////////////
    // Neural network
    ////////////////////////////////////////

    NDList convolution(
            NDArray input,
            NDArray weight,
            NDArray bias,
            Shape stride,
            Shape padding,
            Shape dilation,
            int groups);

    NDList deconvolution(
            NDArray input,
            NDArray weight,
            NDArray bias,
            Shape stride,
            Shape padding,
            Shape outPadding,
            Shape dilation,
            int groups);

    NDList linear(NDArray input, NDArray weight, NDArray bias);

    NDList embedding(NDArray input, NDArray weight, SparseFormat sparse);

    NDList prelu(NDArray input, NDArray alpha);

    NDList dropout(NDArray input, float rate, boolean training);

    NDList layerNorm(NDArray input, Shape normalizedShape, NDArray gamma, NDArray beta, float eps);

    NDList batchNorm(
            NDArray input,
            NDArray runningMean,
            NDArray runningVar,
            NDArray gamma,
            NDArray beta,
            int axis,
            float momentum,
            float eps,
            boolean training);

    /**
     * Applies RNN operation to input data.
     *
     * @param input the inputs to the recurrent operation.
     * @param state the hidden state to the recurrent operation.
     * @param params all params (weights and biases) for the recurrent operation
     * @param hasBiases If false, then the recurrent operation does not use bias weights b_ih and
     *     b_hh
     * @param numLayers the number of recurrent layers.
     * @param activation the activation function to use
     * @param dropRate If non-zero, introduces a Dropout layer on the outputs of each RNN layer
     *     except the last layer, with dropout probability equal to dropout
     * @param training apply dropout if is true
     * @param bidirectional If true, becomes a bidirectional RNN
     * @param batchFirst If true, then the input and output NDArray are provided as (batch, seq,
     *     feature)
     * @return the output of the operation
     */
    NDList rnn(
            NDArray input,
            NDArray state,
            NDList params,
            boolean hasBiases,
            int numLayers,
            RNN.Activation activation,
            double dropRate,
            boolean training,
            boolean bidirectional,
            boolean batchFirst);

    /**
     * Applies GRU operation to input data.
     *
     * @param input the inputs to the GRU operation.
     * @param state the hidden state to the GRU operation.
     * @param params all params (weights and biases) for the GRU operation
     * @param hasBiases If false, then the recurrent operation does not use bias weights b_ih and
     *     b_hh
     * @param numLayers the number of recurrent layers.
     * @param dropRate If non-zero, introduces a Dropout layer on the outputs of each GRU layer
     *     except the last layer, with dropout probability equal to dropout
     * @param training apply dropout if is true
     * @param bidirectional If true, becomes a bidirectional GRU
     * @param batchFirst If true, then the input and output NDArray are provided as (batch, seq,
     *     feature)
     * @return the output of the operation
     */
    NDList gru(
            NDArray input,
            NDArray state,
            NDList params,
            boolean hasBiases,
            int numLayers,
            double dropRate,
            boolean training,
            boolean bidirectional,
            boolean batchFirst);

    /**
     * Applies LSTM operation to input data.
     *
     * @param input the inputs to the LSTM operation.
     * @param states the hidden state and cell state to the LSTM operation.
     * @param params all params (weights and biases) for the LSTM operation
     * @param hasBiases If false, then the recurrent operation does not use bias weights b_ih and
     *     b_hh
     * @param numLayers the number of recurrent layers.
     * @param dropRate If non-zero, introduces a Dropout layer on the outputs of each LSTM layer
     *     except the last layer, with dropout probability equal to dropout
     * @param training apply dropout if is true
     * @param bidirectional If true, becomes a bidirectional LSTM
     * @param batchFirst If true, then the input and output NDArray are provided as (batch, seq,
     *     feature)
     * @return the output of the operation
     */
    NDList lstm(
            NDArray input,
            NDList states,
            NDList params,
            boolean hasBiases,
            int numLayers,
            double dropRate,
            boolean training,
            boolean bidirectional,
            boolean batchFirst);

    ////////////////////////////////////////
    // Image and CV
    ////////////////////////////////////////

    /**
     * Normalizes a NDArray of shape CHW or NCHW with mean and standard deviation.
     *
     * <p>Given mean `(m1, ..., mn)` and std `(s\ :sub:`1`\ , ..., s\ :sub:`n`)` for `n` channels,
     * this transform normalizes each channel of the input tensor with: output[i] = (input[i] - m\
     * :sub:`i`\ ) / s\ :sub:`i`
     *
     * @param mean the mean value for each channel
     * @param std the standard deviation for each channel
     * @return the result of normalization
     */
    default NDArray normalize(float[] mean, float[] std) {
        NDManager manager = getArray().getManager();
        int dim = getArray().getShape().dimension();
        Shape shape = (dim == 3) ? new Shape(3, 1, 1) : new Shape(1, 3, 1, 1);
        try (NDArray meanArr = manager.create(mean, shape);
                NDArray stdArr = manager.create(std, shape)) {
            return getArray().sub(meanArr).divi(stdArr);
        }
    }

    default NDArray toTensor() {
        NDManager manager = getArray().getManager();
        try (NDManager subManager = manager.newSubManager()) {
            NDArray array = getArray();
            array.attach(subManager);

            NDArray result = array;
            int dim = result.getShape().dimension();
            if (dim == 3) {
                result = result.expandDims(0);
            }
            result = result.div(255.0).transpose(0, 3, 1, 2);
            if (dim == 3) {
                result = result.squeeze(0);
            }
            // The network by default takes float32
            if (!result.getDataType().equals(DataType.FLOAT32)) {
                result = result.toType(DataType.FLOAT32, false);
            }
            array.attach(manager);
            result.attach(manager);
            return result;
        }
    }

    NDArray resize(int width, int height, int interpolation);

    default NDArray crop(int x, int y, int width, int height) {
        NDArray array = getArray();
        StringBuilder sb = new StringBuilder(30);
        if (array.getShape().dimension() == 4) {
            sb.append(":,");
        }
        sb.append(y)
                .append(':')
                .append(y + height)
                .append(',')
                .append(x)
                .append(':')
                .append(x + width)
                .append(",:");
        return array.get(sb.toString());
    }

    // TODO: default can be implemented by using np.flip
    NDArray randomFlipLeftRight();

    // TODO: default can be implemented by using np.flip
    NDArray randomFlipTopBottom();

    // TODO: add TorchVision support
    NDArray randomBrightness(float brightness);

    // TODO: add TorchVision support
    NDArray randomHue(float hue);

    // TODO: add TorchVision support
    NDArray randomColorJitter(float brightness, float contrast, float saturation, float hue);

    ////////////////////////////////////////
    // Miscellaneous
    ////////////////////////////////////////

    /**
     * Returns an {@link NDArrayIndexer}.
     *
     * @return an {@link NDArrayIndexer}
     */
    NDArrayIndexer getIndexer();

    /**
     * Returns elements chosen from the {@code NDArray} or the other {@code NDArray} depending on
     * condition.
     *
     * <p>Given three {@code NDArray}s, condition, this, and other, returns an {@code NDArray} with
     * the elements from this or other, depending on whether the elements from condition {@code
     * NDArray} are {@code true} or {@code false}. If condition has the same shape as this, each
     * element in the output {@code NDArray} is from this if the corresponding element in the
     * condition is {@code true}, and from other if {@code false}.
     *
     * <p>Note that all non-zero values are interpreted as {@code true} in condition {@link
     * NDArray}.
     *
     * @param condition the condition {@code NDArray}
     * @param other the other {@code NDArray}
     * @return the result {@code NDArray}
     */
    NDArray where(NDArray condition, NDArray other);

    /**
     * Joins a sequence of {@code NDArray}s in {@link NDList} along a new axis.
     *
     * <p>The axis parameter specifies the index of the new axis in the dimensions of the result.
     * For example, if axis=0 it will be the first dimension and if axis=-1 it will be the last
     * dimension.
     *
     * @param arrays the input {@link NDList}. Each {@code NDArray} in the {@link NDList} must have
     *     the same shape as the {@code NDArray}
     * @param axis the axis in the result {@code NDArray} along which the input {@link NDList} are
     *     stacked
     * @return the result {@code NDArray}. The stacked {@code NDArray} has one more dimension than
     *     the the {@code NDArray}
     */
    NDArray stack(NDList arrays, int axis);

    /**
     * Joins a sequence of {@code NDArray}s in {@link NDList} along first axis.
     *
     * @param arrays the input {@link NDList}. Each {@code NDArray} in the {@link NDList} must have
     *     the same shape as the {@code NDArray}
     * @return the result {@code NDArray}. The stacked {@code NDArray} has one more dimension than
     *     the {@code NDArray}s in {@link NDList}
     */
    default NDArray stack(NDList arrays) {
        return stack(arrays, 0);
    }

    /**
     * Joins a {@link NDList} along an existing axis.
     *
     * @param arrays a {@link NDList} which have the same shape as the {@code NDArray}, except in
     *     the dimension corresponding to axis
     * @param axis the axis along which the {@link NDList} will be joined
     * @return the concatenated {@code NDArray}
     */
    NDArray concat(NDList arrays, int axis);

    /**
     * Joins a {@link NDList} along first axis.
     *
     * @param arrays a {@link NDList} which have the same shape as the {@code NDArray}, except in
     *     the dimension corresponding to axis
     * @return the concatenated {@code NDArray}
     */
    default NDArray concat(NDList arrays) {
        return concat(arrays, 0);
    }

    /**
     * Computes Multibox training targets.
     *
     * @param inputs a NDList of (anchors, labels, and class prediction)
     * @param iouThreshold the anchor-GroundTruth overlap threshold to be regarded as a positive
     *     match
     * @param ignoreLabel the label for ignored anchors
     * @param negativeMiningRatio the max negative to positive samples ratio, use -1 to disable
     *     mining
     * @param negativeMiningThreshold the threshold used for negative mining
     * @param minNegativeSamples the minimum number of negative samples
     * @return an NDList of (bounding box labels, bounding box masks, class labels)
     */
    NDList multiBoxTarget(
            NDList inputs,
            float iouThreshold,
            float ignoreLabel,
            float negativeMiningRatio,
            float negativeMiningThreshold,
            int minNegativeSamples);

    /**
     * Generate prior(anchor) boxes from data, sizes and ratios.
     *
     * @param sizes List of sizes of generated MultiBoxPriores
     * @param ratios List of aspect ratios of generated MultiBoxPriores
     * @param steps Priorbox step across y and x, -1 for auto calculation
     * @param offsets Priorbox center offsets, y and x respectively
     * @param clip Whether to clip out-of-boundary boxes
     * @return an NDList of anchor boxes
     */
    NDList multiBoxPrior(
            List<Float> sizes,
            List<Float> ratios,
            List<Float> steps,
            List<Float> offsets,
            boolean clip);

    /**
     * Converts multi-box detection predictions.
     *
     * @param inputs a NDList of (anchors, labels, and class prediction) in that order
     * @param clip whether to clip out-of-boundary boxes
     * @param threshold the threshold to be a positive prediction
     * @param backgroundId the background id
     * @param nmsThreshold the non-maximum suppression threshold
     * @param forceSuppress whether to suppress all detections regardless of class_id
     * @param nmsTopK the number of detections to keep before NMS, -1 for no limit
     * @return an NDList
     */
    NDList multiBoxDetection(
            NDList inputs,
            boolean clip,
            float threshold,
            int backgroundId,
            float nmsThreshold,
            boolean forceSuppress,
            int nmsTopK);

    /**
     * Get internal {@link NDArray}.
     *
     * @return a NDArray
     */
    NDArray getArray();
}
