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
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.NDUtils;
import ai.djl.ndarray.index.NDArrayIndexer;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.internal.NDArrayEx;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.ndarray.types.SparseFormat;
import ai.djl.nn.recurrent.RNN;
import ai.djl.pytorch.jni.JniUtils;
import ai.djl.util.Pair;
import ai.djl.util.PairList;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

/** {@code PtNDArrayEx} is the PyTorch implementation of the {@link NDArrayEx}. */
public class PtNDArrayEx implements NDArrayEx {

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
    public PtNDArray rdivi(NDArray b) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray rmodi(NDArray b) {
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
            float learningRateBiasCorrection,
            float weightDecay,
            float rescaleGrad,
            float clipGrad,
            float beta1,
            float beta2,
            float epsilon,
            boolean lazyUpdate,
            boolean adamw) {
        // TODO: Lazy update not used
        PtNDManager manager = array.getManager();
        JniUtils.adamUpdate(
                manager.from(inputs.get(0)),
                manager.from(inputs.get(1)),
                manager.from(inputs.get(2)),
                manager.from(inputs.get(3)),
                learningRate,
                learningRateBiasCorrection,
                weightDecay,
                rescaleGrad,
                clipGrad,
                beta1,
                beta2,
                epsilon,
                adamw);
        // call zero-grad
        JniUtils.zeroGrad(manager.from(weights.singletonOrThrow()));
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
        PtNDManager manager = array.getManager();
        JniUtils.sgdUpdate(
                manager.from(inputs.get(0)),
                manager.from(inputs.get(1)),
                (momentum == 0f) ? null : manager.from(inputs.get(2)),
                learningRate,
                weightDecay,
                rescaleGrad,
                clipGrad,
                momentum);
        // call zero-grad
        JniUtils.zeroGrad(manager.from(weights.singletonOrThrow()));
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
        PtNDManager manager = array.getManager();
        return new NDList(
                JniUtils.convolution(
                        manager.from(input),
                        manager.from(weight),
                        manager.from(bias),
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
        PtNDManager manager = array.getManager();
        return new NDList(
                JniUtils.linear(manager.from(input), manager.from(weight), manager.from(bias)));
    }

    /** {@inheritDoc} */
    @Override
    public NDList embedding(NDArray input, NDArray weight, SparseFormat sparseFormat) {
        if (!sparseFormat.equals(SparseFormat.DENSE) && !sparseFormat.equals(SparseFormat.COO)) {
            throw new IllegalArgumentException("PyTorch only supports COO");
        }
        PtNDManager manager = array.getManager();
        return new NDList(
                JniUtils.embedding(
                        manager.from(input),
                        manager.from(weight),
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
        PtNDManager manager = array.getManager();
        return new NDList(JniUtils.dropout(manager.from(input), rate, training));
    }

    /** {@inheritDoc} */
    @Override
    public NDList layerNorm(
            NDArray input, Shape normalizedShape, NDArray gamma, NDArray beta, float eps) {
        PtNDManager manager = array.getManager();
        return new NDList(
                JniUtils.layerNorm(
                        manager.from(input),
                        normalizedShape,
                        manager.from(gamma),
                        manager.from(beta),
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
        PtNDManager manager = array.getManager();
        if (axis == -1) {
            return new NDList(
                    JniUtils.batchNorm(
                            manager.from(input),
                            manager.from(runningMean),
                            manager.from(runningVar),
                            manager.from(gamma),
                            manager.from(beta),
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
                            manager.from(result),
                            manager.from(runningMean),
                            manager.from(runningVar),
                            manager.from(gamma),
                            manager.from(beta),
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
        PtNDManager manager = array.getManager();
        return JniUtils.rnn(
                manager.from(input),
                manager.from(state),
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
        PtNDManager manager = array.getManager();
        return JniUtils.gru(
                manager.from(input),
                manager.from(state),
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
                array.getManager().from(input),
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
    public NDArray interpolation(long[] size, int mode, boolean alignCorners) {
        return JniUtils.interpolate(
                array.getManager().from(array), size, getInterpolationMode(mode), false);
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray resize(int width, int height, int interpolation) {
        // create subManager to help close intermediate NDArray
        PtNDManager manager = array.getManager();
        try (NDManager subManager = manager.newSubManager()) {
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
            // change from HWC to CHW order
            result = result.transpose(0, 3, 1, 2);
            result =
                    JniUtils.interpolate(
                                    array.getManager().from(result),
                                    new long[] {height, width},
                                    getInterpolationMode(interpolation),
                                    false)
                            .transpose(0, 2, 3, 1);
            if (dim == 3) {
                result = result.squeeze(0);
            }
            result = result.toType(array.getDataType(), false);
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
    public NDArrayIndexer getIndexer(NDManager manager) {
        return new PtNDArrayIndexer((PtNDManager) manager);
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray where(NDArray condition, NDArray other) {
        // Try to broadcast if shape mismatch
        if (!condition.getShape().equals(array.getShape())) {
            throw new UnsupportedOperationException(
                    "condition and self shape mismatch, broadcast is not supported");
        }
        PtNDManager manager = array.getManager();
        return JniUtils.where(manager.from(condition), array, manager.from(other));
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray stack(NDList arrays, int axis) {
        PtNDArray[] srcArray = new PtNDArray[arrays.size() + 1];
        srcArray[0] = array;
        int i = 1;
        PtNDManager manager = array.getManager();
        for (NDArray arr : arrays) {
            srcArray[i++] = manager.from(arr);
        }
        return JniUtils.stack(srcArray, axis);
    }

    /** {@inheritDoc} */
    @Override
    public PtNDArray concat(NDList list, int axis) {
        NDUtils.checkConcatInput(list);

        PtNDArray[] srcArray = new PtNDArray[list.size() + 1];
        srcArray[0] = array;
        int i = 1;
        PtNDManager manager = array.getManager();
        for (NDArray arr : list) {
            srcArray[i++] = manager.from(arr);
        }
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

        float[] variances = {0.1f, 0.1f, 0.2f, 0.2f}; // from mxnet default

        NDArray anchors = inputs.get(0);
        NDArray labels = inputs.get(1);
        NDArray preds = inputs.get(2);

        long numBatches = labels.size(0);
        long numLabels = labels.size(1);
        long numAnchors = anchors.size(1);

        NDArray locTarget = inputs.getManager().create(new Shape(numBatches, numAnchors * 4));
        NDArray classTarget = inputs.getManager().create(new Shape(numBatches, numAnchors));
        NDArray locMask = inputs.getManager().create(new Shape(numBatches, numAnchors * 4));

        for (int batchInd = 0; batchInd < numBatches; batchInd++) {

            int numValidGT = 0;
            for (int labelInd = 0; labelInd < numLabels; labelInd++) {
                if (labels.getFloat(batchInd, labelInd, 0) == -1 &&
                        labels.getFloat(batchInd, labelInd, 1) == -1 &&
                        labels.getFloat(batchInd, labelInd, 2) == -1 &&
                        labels.getFloat(batchInd, labelInd, 3) == -1 &&
                        labels.getFloat(batchInd, labelInd, 4) == -1) {
                    break;
                }
                numValidGT++;
            }
            if (numValidGT > 0) {
                Boolean[] gtFlags = new Boolean[numValidGT];
                Arrays.fill(gtFlags, false);
                PairList<Float, Integer> maxMatches = new PairList<>();
                for (int i = 0; i < numAnchors; i++) {
                    maxMatches.add(new Pair<>(-1f, -1));
                }
                int[] anchorFlags = new int[Math.toIntExact(numAnchors)];
                Arrays.fill(anchorFlags, -1);
                int numPositive = 0;

                // while (std::find(gt_flags.begin(), gt_flags.end(), false) != gt_flags.end()) {
                while (Arrays.stream(gtFlags).anyMatch(b -> !b)) {
                    int bestAnchor = -1;
                    int bestGT = -1;
                    float maxOverlap = 1e-6f;
                    for (int j = 0; j < numAnchors; j++) {
                        if (anchorFlags[j] == 1) {
                            continue; // already matched
                        }
                        for (int k = 0; k < numValidGT; k++) {
                            if (gtFlags[k]) {
                                continue; // already matched
                            }
                            // anchor index is j, label index is k
                            float[] outputsKRow4 = anchors.get(new NDIndex(String.format("%d,%d,0:4", 0, j))).toFloatArray();
                            float[] outputsJRow4 = labels.get(new NDIndex(String.format("%d,%d,1:5", batchInd, k))).toFloatArray();
                            float iou = calculateOverlap(outputsKRow4, outputsJRow4);

                            if (iou > maxOverlap) {
                                bestAnchor = j;
                                bestGT     = k;
                                maxOverlap = iou;
                            }
                        }
                    }
                    if (bestAnchor == -1) {
                        assert bestGT == -1;
                        break;
                    } else {
                        assert maxMatches.get(bestAnchor).getKey() == -1f;
                        assert maxMatches.get(bestAnchor).getValue() == -1;
                        maxMatches.keys().set(bestAnchor, maxOverlap);
                        maxMatches.values().set(bestAnchor, bestGT);
                        numPositive++;
                        gtFlags[bestGT] = true;
                        anchorFlags[bestAnchor] = 1;
                    }
                }
                if (iouThreshold > 0) {
                    for (int j = 0; j < numAnchors; j++) {
                        if (anchorFlags[j] == 1) {
                            continue; // already matched
                        }
                        int bestGT = -1;
                        float maxIou = -1f;
                        for (int k = 0; k < numValidGT; k++) {
                            // anchor index is j, label index is k
                            float[] outputsKRow4 = anchors.get(new NDIndex(String.format("%d,%d,0:4", 0, j))).toFloatArray();
                            float[] outputsJRow4 = labels.get(new NDIndex(String.format("%d,%d,1:5", batchInd, k))).toFloatArray();
                            float iou = calculateOverlap(outputsKRow4, outputsJRow4);

                            if (iou > maxIou) {
                                bestGT = k;
                                maxIou = iou;
                            }
                        }
                        if (bestGT != -1) {
                            assert maxMatches.get(j).getKey() == -1f;
                            assert maxMatches.get(j).getValue() == -1;
                            maxMatches.keys().set(j, maxIou);
                            maxMatches.values().set(j, bestGT);
                            if (maxIou > iouThreshold) {
                                numPositive++;
                                gtFlags[bestGT] = true;
                                anchorFlags[j] = 1;
                            }
                        }
                    }
                }

                if (negativeMiningRatio > 0) {
                    long numClasses = preds.size(1);
                    assert negativeMiningThreshold > 0;
                    int numNegative = (int) (numPositive * negativeMiningRatio);
                    if (numNegative > (numAnchors - numPositive)) {
                        numNegative = (int) (numAnchors - numPositive);
                    }
                    if (numNegative > 0) {
                        // use negative mining, pick up "best" negative samples
                        List<Pair<Double, Integer>> pairs = new ArrayList<>(); // value, index
                        for (int j = 0; j < numAnchors; j++) {
                            if (anchorFlags[j] == 1) {
                                continue; // already matched
                            }
                            if (maxMatches.get(j).getKey() == -1) { // not calculated yet
                                int bestGT = -1;
                                float maxIou = -1f;
                                for (int k = 0; k < numValidGT; k++) {
                                    float[] outputsKRow4 = anchors.get(new NDIndex(String.format("%d,%d,0:4", 0, j))).toFloatArray();
                                    float[] outputsJRow4 = labels.get(new NDIndex(String.format("%d,%d,1:5", batchInd, k))).toFloatArray();
                                    float iou = calculateOverlap(outputsKRow4, outputsJRow4);

                                    if (iou > maxIou) {
                                        bestGT = k;
                                        maxIou = iou;
                                    }
                                }
                                if (bestGT != -1) {
                                    assert maxMatches.get(j).getKey() == -1f;
                                    assert maxMatches.get(j).getValue() == -1;
                                    maxMatches.keys().set(j, maxIou);
                                    maxMatches.values().set(j, bestGT);
                                }
                            }
                            if (maxMatches.get(j).getKey() < negativeMiningThreshold && anchorFlags[j] == -1) {
                                float maxVal = preds.getFloat(batchInd, j, 0);
                                for (int k = 1; k < numClasses; k++) {
                                    float tmp = preds.getFloat(batchInd, j, k);
                                    if (tmp > maxVal)
                                        maxVal = tmp;
                                }
                                double sum = 0;
                                for (int k = 0; k < numClasses; k++) {
                                    float tmp = preds.getFloat(batchInd, j, k);
                                    sum += Math.exp(tmp - maxVal);
                                }
                                double prob = Math.exp(preds.getFloat(batchInd, j, 0) - maxVal) / sum;
                                // loss should be -log(x), but value does not matter, skip log
                                pairs.add(new Pair<>(-prob, j));
                            }
                        }
                        assert pairs.size() >= numNegative;
                        pairs.sort(Comparator.comparingDouble(Pair::getKey));
                        for (int i = 0; i < numNegative; i++) {
                            anchorFlags[pairs.get(i).getValue()] = 0;
                        }
                    }
                } else {
                    // use all negative samples
                    for (int i = 0; i < numAnchors; ++i) {
                        if (anchorFlags[i] != 1) {
                            anchorFlags[i] = 0;
                        }
                    }
                }

                // assign training targets
                for (int i = 0; i < numAnchors; i++) {
                    if (anchorFlags[i] == 1) {
                        // positive sample
                        assert maxMatches.get(i).getValue() >= 0;
                        // 0 reserved for background
                        classTarget.set(
                                new NDIndex(batchInd, i),
                                labels.getFloat(batchInd, maxMatches.get(i).getValue(), 0) + 1
                        );
                        locMask.set(new NDIndex(batchInd + "," + (i*4) + ":" + ((i*4)+4)), 1);
                        // todo anchor zero index...?
                        AssignLocTargets(
                                anchors.get(new NDIndex(0 + "," + i + ",:")),
                                labels.get(new NDIndex(batchInd + "," + maxMatches.get(i).getValue() + ",1:")),
                                locTarget.get(new NDIndex(String.format("%d,%d:%d", batchInd, i * 4, (i * 4) + 4))),
                                variances
                        );
                    } else if (anchorFlags[i] == 0) {
                        // negative sample
                        classTarget.set(new NDIndex(batchInd, i), 0);
                        locMask.set(new NDIndex(batchInd + "," + (i*4) + ":" + ((i*4)+4)), 0);
                    }
                }
            }
        }
        return new NDList(locTarget, locMask, classTarget);
    }

    void AssignLocTargets(NDArray anchor,
                          NDArray label,
                          NDArray locTarget,
                          float[] variances) {
        float al   = anchor.getFloat(0);
        float at   = anchor.getFloat(1);
        float ar   = anchor.getFloat(2);
        float ab   = anchor.getFloat(3);
        float aw   = ar - al;
        float ah   = ab - at;
        float ax   = (float) ((al + ar) * 0.5);
        float ay   = (float) ((at + ab) * 0.5);
        float gl   = label.getFloat(0);
        float gt   = label.getFloat(1);
        float gr   = label.getFloat(2);
        float gb   = label.getFloat(3);
        float gw   = gr - gl;
        float gh   = gb - gt;
        float gx   = (float) ((gl + gr) * 0.5);
        float gy   = (float) ((gt + gb) * 0.5);
        locTarget.set(new NDIndex(0), (gx - ax) / aw / variances[0]);
        locTarget.set(new NDIndex(1), (gy - ay) / ah / variances[1]);
        locTarget.set(new NDIndex(2), Math.log(gw / aw) / variances[2]);
        locTarget.set(new NDIndex(3), Math.log(gh / ah) / variances[3]);
    }


    /** {@inheritDoc} */
    @Override
    public NDList multiBoxPrior(
            List<Float> sizes,
            List<Float> ratios,
            List<Float> steps,
            List<Float> offsets,
            boolean clip) {

        NDManager ndManager = array.getManager();

        Shape input = array.getShape();
        int inHeight = Math.toIntExact(input.get(2));
        int inWidth = Math.toIntExact(input.get(3));

        if (steps.get(0) <= 0 || steps.get(1) <= 0) {
            // estimate using layer shape
            steps.set(0, 1.f / inHeight);
            steps.set(1, 1.f / inWidth);
        }

        float stepX = steps.get(1);
        float stepY = steps.get(0);
        int numSizes = sizes.size();
        int numRatios = ratios.size();
        int count = 0;

        float[][] out = new float[inHeight * inWidth * numSizes * 2][4];

        for (int r = 0; r < inHeight; ++r) {
            float centerY = (r + offsets.get(0)) * stepY;
            for (int c = 0; c < inWidth; ++c) {
                float centerX = (c + offsets.get(1)) * stepX;
                // ratio = first ratio, various sizes
                float ratio = numRatios > 0 ? (float) Math.sqrt(ratios.get(0)) : 1.f;
                for (int i = 0; i < numSizes; ++i) {
                    float size = sizes.get(i);
                    float w = size * inHeight / inWidth * ratio / 2;
                    float h = size / ratio / 2;

                    out[count][0] = centerX - w; // xmin
                    out[count][1] = centerY - h; // ymin
                    out[count][2] = centerX + w; // xmax
                    out[count][3] = centerY + h; // ymax
                    ++count;
                }
                // various ratios, size = min_size = size[0]
                float size = sizes.get(0);
                for (int j = 1; j < numRatios; ++j) {
                    float ratioLocal = (float) Math.sqrt(ratios.get(j));
                    float w = size * inHeight / inWidth * ratioLocal / 2;
                    float h = size / ratioLocal / 2;
                    out[count][0] = centerX - w; // xmin
                    out[count][1] = centerY - h; // ymin
                    out[count][2] = centerX + w; // xmax
                    out[count][3] = centerY + h; // ymax
                    ++count;
                }
            }
        }
        NDArray ndArray = ndManager.create(out).expandDims(0);
        return new NDList(ndArray);
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
        assert (inputs.size() == 3);

        NDArray clsProb = inputs.get(0);
        NDArray locPred = inputs.get(1);
        NDArray anchors = inputs.get(2).reshape(new Shape(-1, 4));

        NDManager ndManager = array.getManager();

        NDArray variances = ndManager.create(new float[] {0.1f, 0.1f, 0.2f, 0.2f});

        assert (variances.size() == 4); // << "Variance size must be 4";
        final int numClasses = (int) clsProb.size(1);
        final int numAnchors = (int) clsProb.size(2);
        final int numBatches = (int) clsProb.size(0);

        final float[] pAnchor = anchors.toFloatArray();

        // [id, prob, xmin, ymin, xmax, ymax]
        // TODO Move to NDArray-based implementation
        NDList batchOutputs = new NDList();
        for (int nbatch = 0; nbatch < numBatches; ++nbatch) {
            float[][] outputs = new float[numAnchors][6];
            final float[] pClsProb = clsProb.get(nbatch).toFloatArray();
            final float[] pLocPred = locPred.get(nbatch).toFloatArray();

            for (int i = 0; i < numAnchors; ++i) {
                // find the predicted class id and probability
                float score = -1;
                int id = 0;
                for (int j = 1; j < numClasses; ++j) {
                    float temp = pClsProb[j * numAnchors + i];
                    if (temp > score) {
                        score = temp;
                        id = j;
                    }
                }

                if (id > 0 && score < threshold) {
                    id = 0;
                }

                // [id, prob, xmin, ymin, xmax, ymax]
                outputs[i][0] = id - 1;
                outputs[i][1] = score;
                int offset = i * 4;
                float[] pAnchorRow4 = new float[4];
                pAnchorRow4[0] = pAnchor[offset];
                pAnchorRow4[1] = pAnchor[offset + 1];
                pAnchorRow4[2] = pAnchor[offset + 2];
                pAnchorRow4[3] = pAnchor[offset + 3];
                float[] pLocPredRow4 = new float[4];
                pLocPredRow4[0] = pLocPred[offset];
                pLocPredRow4[1] = pLocPred[offset + 1];
                pLocPredRow4[2] = pLocPred[offset + 2];
                pLocPredRow4[3] = pLocPred[offset + 3];
                float[] outRowLast4 =
                        transformLocations(
                                pAnchorRow4,
                                pLocPredRow4,
                                clip,
                                variances.toFloatArray()[0],
                                variances.toFloatArray()[1],
                                variances.toFloatArray()[2],
                                variances.toFloatArray()[3]);
                outputs[i][2] = outRowLast4[0];
                outputs[i][3] = outRowLast4[1];
                outputs[i][4] = outRowLast4[2];
                outputs[i][5] = outRowLast4[3];
            }

            outputs =
                    Arrays.stream(outputs)
                            .filter(o -> o[0] >= 0)
                            .sorted(Comparator.comparing(o -> -o[1]))
                            .toArray(float[][]::new);

            // apply nms
            for (int i = 0; i < outputs.length; ++i) {
                for (int j = i + 1; j < outputs.length; ++j) {
                    if (outputs[i][0] == outputs[j][0]) {
                        float[] outputsIRow4 = new float[4];
                        float[] outputsJRow4 = new float[4];
                        outputsIRow4[0] = outputs[i][2];
                        outputsIRow4[1] = outputs[i][3];
                        outputsIRow4[2] = outputs[i][4];
                        outputsIRow4[3] = outputs[i][5];
                        outputsJRow4[0] = outputs[j][2];
                        outputsJRow4[1] = outputs[j][3];
                        outputsJRow4[2] = outputs[j][4];
                        outputsJRow4[3] = outputs[j][5];
                        float iou = calculateOverlap(outputsIRow4, outputsJRow4);
                        if (iou >= nmsThreshold) {
                            outputs[j][0] = -1;
                        }
                    }
                }
            }
            batchOutputs.add(ndManager.create(outputs));
        } // end iter batch

        NDArray pOutNDArray = NDArrays.stack(batchOutputs);
        NDList resultNDList = new NDList();
        resultNDList.add(pOutNDArray);
        assert (resultNDList.size() == 1);
        return resultNDList;
    }

    private float[] transformLocations(
            final float[] anchors,
            final float[] locPred,
            final boolean clip,
            final float vx,
            final float vy,
            final float vw,
            final float vh) {
        float[] outRowLast4 = new float[4];
        // transform predictions to detection results
        float al = anchors[0];
        float at = anchors[1];
        float ar = anchors[2];
        float ab = anchors[3];
        float aw = ar - al;
        float ah = ab - at;
        float ax = (al + ar) / 2.f;
        float ay = (at + ab) / 2.f;
        float px = locPred[0];
        float py = locPred[1];
        float pw = locPred[2];
        float ph = locPred[3];
        float ox = px * vx * aw + ax;
        float oy = py * vy * ah + ay;
        float ow = (float) (Math.exp(pw * vw) * aw / 2);
        float oh = (float) (Math.exp(ph * vh) * ah / 2);
        outRowLast4[0] = clip ? Math.max(0f, Math.min(1f, ox - ow)) : (ox - ow);
        outRowLast4[1] = clip ? Math.max(0f, Math.min(1f, oy - oh)) : (oy - oh);
        outRowLast4[2] = clip ? Math.max(0f, Math.min(1f, ox + ow)) : (ox + ow);
        outRowLast4[3] = clip ? Math.max(0f, Math.min(1f, oy + oh)) : (oy + oh);
        return outRowLast4;
    }

    private float calculateOverlap(final float[] a, final float[] b) {
        float w = Math.max(0f, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
        float h = Math.max(0f, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
        float i = w * h;
        float u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i;
        return u <= 0.f ? 0f : (i / u);
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
