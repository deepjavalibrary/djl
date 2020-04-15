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
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.pooling.PoolingConvention;
import ai.djl.util.PairList;
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

    NDArray softrelu();

    NDArray softsign();

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

    NDArray maxPool(Shape kernel, Shape stride, Shape pad, PoolingConvention poolingConvention);

    NDArray globalMaxPool();

    NDArray sumPool(Shape kernel, Shape stride, Shape pad, PoolingConvention poolingConvention);

    NDArray globalSumPool();

    NDArray avgPool(
            Shape kernel,
            Shape stride,
            Shape pad,
            PoolingConvention poolingConvention,
            boolean countIncludePad);

    NDArray globalAvgPool();

    NDArray lpPool(
            Shape kernel, Shape stride, Shape pad, PoolingConvention poolingConvention, int pValue);

    NDArray globalLpPool(int pValue);

    ////////////////////////////////////////
    // Optimizer
    ////////////////////////////////////////

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

    /**
     * Computes N-D convolution on (N+2)-D input.
     *
     * @param inputs the inputs to the convolution operation. Msut include input data, weight
     *     parameter matrix, and bias parameter
     * @param kernel the convolution kernel size: (w,), (h, w) or (d, h, w)
     * @param stride the convolution stride: (w,), (h, w) or (d, h, w). Defaults to 1 for each
     *     dimension
     * @param pad the zero pad for convolution: (w,), (h, w) or (d, h, w). Defaults to no padding
     * @param dilate the convolution dilate: (w,), (h, w) or (d, h, w). Defaults to 1 for each
     *     dimension
     * @param numFilters the convolution filter(channel) number
     * @param numGroups the number of group partitions. Defaults to 1
     * @param layout the layout for input, output and weight. Empty for default layout: NCW for 1d,
     *     NCHW for 2d and NCDHW for 3d. NHWC and NDHWC are only supported on GPU
     * @param noBias whether to disable bias parameter. Defaults to false
     * @param additional additional parameters
     * @return the output of the convolution operation
     */
    NDList convolution(
            NDList inputs,
            Shape kernel,
            Shape stride,
            Shape pad,
            Shape dilate,
            int numFilters,
            int numGroups,
            String layout,
            boolean noBias,
            PairList<String, Object> additional);

    NDList fullyConnected(
            NDList inputs,
            long outChannels,
            boolean flatten,
            boolean noBias,
            PairList<String, Object> additional);

    NDList embedding(
            NDList inputs,
            int numItems,
            int embeddingSize,
            boolean sparseGrad,
            DataType dataType,
            PairList<String, Object> additional);

    NDList prelu(NDList inputs, PairList<String, Object> additional);

    NDList dropout(
            NDList inputs,
            float probability,
            int[] sharedAxes,
            PairList<String, Object> additional);

    NDList batchNorm(
            NDList inputs,
            float epsilon,
            float momentum,
            int axis,
            boolean center,
            boolean scale,
            PairList<String, Object> additional);

    /**
     * Applies recurrent layers to input data. Currently, vanilla RNN, LSTM and GRU are implemented,
     * with both multi-layer and bidirectional support.
     *
     * @param inputs the inputs to the recurrent operation. Must include input data, parameter
     *     vector of all trainable parameters concatenated, initial hidden state of the RNN. For
     *     LSTM, it must include initial cell state. If useSequenceLength is true, it must also
     *     include vector of valid sequence lengths for each element in the batch
     * @param mode the type of RNN to compute
     * @param stateSize the sizes of the state for each layer
     * @param dropRate the drop rate of the dropout on the outputs of each RNN layer, except the
     *     last layer
     * @param numStackedLayers the number of stacked layers
     * @param useSequenceLength if set to true, this layer takes in an extra input parameter
     *     sequence_length to specify variable length sequence.
     * @param useBidirectional whether to use bidirectional recurrent layers
     * @param stateOutputs whether to include the state in the output
     * @param additional additional parameters
     * @return the output of the operation
     */
    NDList rnn(
            NDList inputs,
            String mode,
            long stateSize,
            float dropRate,
            int numStackedLayers,
            boolean useSequenceLength,
            boolean useBidirectional,
            boolean stateOutputs,
            PairList<String, Object> additional);

    /**
     * Applies LSTM recurrent layers to input data.
     *
     * @param inputs the inputs to the recurrent operation. Must include input data, parameter
     *     vector of all trainable parameters concatenated, initial hidden state of the RNN and
     *     initial cell state. If useSequenceLength is true, it must also include vector of valid
     *     sequence lengths for each element in the batch
     * @param stateSize the sizes of the state for each layer
     * @param dropRate the drop rate of the dropout on the outputs of each RNN layer, except the
     *     last layer
     * @param numStackedLayers the number of stacked layers
     * @param useSequenceLength if set to true, this layer takes in an extra input parameter
     *     sequence_length to specify variable length sequence.
     * @param useBidirectional whether to use bidirectional recurrent layers
     * @param stateOutputs whether to include the state in the output
     * @param lstmStateClipMin the minimum clip value of LSTM states
     * @param lstmStateClipMax the maximum clip value of LSTM states
     * @param additional additional parameters
     * @return the output of the operation
     */
    NDList lstm(
            NDList inputs,
            long stateSize,
            float dropRate,
            int numStackedLayers,
            boolean useSequenceLength,
            boolean useBidirectional,
            boolean stateOutputs,
            double lstmStateClipMin,
            double lstmStateClipMax,
            PairList<String, Object> additional);

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
        NDArray array = getArray();
        int dim = array.getShape().dimension();
        if (dim == 3) {
            array = array.expandDims(0);
        }
        array = array.div(255.0).transpose(0, 3, 1, 2);
        if (dim == 3) {
            array = array.squeeze(0);
        }
        // The network by default takes float32
        return (!array.getDataType().equals(DataType.FLOAT32))
                ? array.toType(DataType.FLOAT32, false)
                : array;
    }

    NDArray resize(int width, int height);

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

    ////////////////////////////////////////
    // Miscellaneous
    ////////////////////////////////////////

    /**
     * Picks elements from an input array according to the input indices along the given axis.
     *
     * @param index the index array
     * @param axis the axis used to pick the elements. Negative values means indexing happens from
     *     right to left. If it is `None`, the elements in the index w.r.t the flattened input will
     *     be picked.
     * @param keepDims If true, the axis where we pick the elements is left in the result as
     *     dimension with size one.
     * @param mode Specify how out-of-bound indices behave. "clip" means clip to the range. So, if
     *     all indices mentioned are too large, they are replaced by the index that addresses the
     *     last element along an axis. "wrap" means to wrap around.
     * @return a copy of the array
     */
    NDArray pick(NDArray index, int axis, boolean keepDims, String mode);

    /**
     * Picks elements from an input array according to the input indices along the given axis.
     *
     * @param index the index array
     * @param axis the axis used to pick the elements. Negative values mean indexing happen from
     *     right to left. If it is `None`, the elements in the index w.r.t the flattened input will
     *     be picked.
     * @param keepDims If true, the axis where we pick the elements is left in the result as
     *     dimension with size one.
     * @return a copy of the array
     */
    default NDArray pick(NDArray index, int axis, boolean keepDims) {
        return pick(index, axis, keepDims, "clip");
    }

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
