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
package ai.djl.nn.transformer;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.training.loss.Loss;

/** Calculates the loss for the next sentence prediction task. */
public class BertNextSentenceLoss extends Loss {

    private int labelIdx;

    private int nextSentencePredictionIdx;

    /**
     * Creates a new bert next sentence loss.
     *
     * @param labelIdx index of the next sentence labels
     * @param nextSentencePredictionIdx index of the next sentence prediction in the bert output
     */
    public BertNextSentenceLoss(int labelIdx, int nextSentencePredictionIdx) {
        super("BertNSLoss");
        this.labelIdx = labelIdx;
        this.nextSentencePredictionIdx = nextSentencePredictionIdx;
    }

    @Override
    public NDArray evaluate(NDList labels, NDList predictions) {
        try (NDManager scope = NDManager.subManagerOf(labels)) {
            scope.tempAttachAll(labels, predictions);
            NDArray label = labels.get(labelIdx).toType(DataType.FLOAT32, false);
            // predictions are log(softmax)
            NDArray logPredictions = predictions.get(nextSentencePredictionIdx);
            NDArray oneHotLabels = label.oneHot(2);
            // we use negative log likelihood as loss: log(softmax) turns high confidence into
            // negative values near one, low confidence into negative values near -inf,
            // negating gives almost 0 for high confidence and near +inf for very low confidence
            NDArray logPredictionForLabels = oneHotLabels.mul(logPredictions);
            NDArray summedPredictions = logPredictionForLabels.sum(new int[] {1});
            NDArray perExampleLoss = summedPredictions.mul(-1f);
            NDArray result = perExampleLoss.mean();
            return scope.ret(result);
        }
    }

    /**
     * Calculates the fraction of correct predictions.
     *
     * @param labels the labels with the correct predictions
     * @param predictions the bert pretraining model output
     * @return the fraction of correct predictions.
     */
    public NDArray accuracy(NDList labels, NDList predictions) {
        try (NDManager scope = NDManager.subManagerOf(labels)) {
            scope.tempAttachAll(labels, predictions);
            NDArray label = labels.get(labelIdx);
            NDArray predictionLogProbs = predictions.get(nextSentencePredictionIdx);
            // predictions are log(softmax) -> highest confidence is highest (negative) value near 0
            NDArray prediction = predictionLogProbs.argMax(1).toType(DataType.INT32, false);
            NDArray equalCount = label.eq(prediction).sum().toType(DataType.FLOAT32, false);
            NDArray result = equalCount.div(label.getShape().size());

            return scope.ret(result);
        }
    }
}
