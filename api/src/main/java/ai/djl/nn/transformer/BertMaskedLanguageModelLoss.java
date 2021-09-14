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

/** The loss for the bert masked language model task. */
public class BertMaskedLanguageModelLoss extends Loss {

    private int labelIdx;
    private int maskIdx;
    private int logProbsIdx;

    /**
     * Creates an MLM loss.
     *
     * @param labelIdx index of labels
     * @param maskIdx index of mask
     * @param logProbsIdx index of log probs
     */
    public BertMaskedLanguageModelLoss(int labelIdx, int maskIdx, int logProbsIdx) {
        super("BertMLLoss");
        this.labelIdx = labelIdx;
        this.maskIdx = maskIdx;
        this.logProbsIdx = logProbsIdx;
    }

    @Override
    public NDArray evaluate(NDList labels, NDList predictions) {
        try (NDManager scope = NDManager.subManagerOf(labels)) {
            scope.tempAttachAll(labels, predictions);

            NDArray logProbs = predictions.get(logProbsIdx); // (B * I, D)
            int dictionarySize = (int) logProbs.getShape().get(1);
            NDArray targetIds = labels.get(labelIdx).flatten(); // (B * I)
            NDArray mask = labels.get(maskIdx).flatten().toType(DataType.FLOAT32, false); // (B * I)
            NDArray targetOneHots = targetIds.oneHot(dictionarySize);
            // Multiplying log_probs and one_hot_labels leaves the log probabilities of the correct
            // entries.
            // By summing we get the total predicition quality. We want to minimize the error,
            // so we negate the value - as we have logarithms, probability = 1 means log(prob) = 0,
            // the less sure we are the smaller the log value.
            NDArray perExampleLoss = logProbs.mul(targetOneHots).sum(new int[] {1}).mul(-1);
            // Multiplying log_probs and one_hot_labels leaves the log probabilities of the correct
            // entries.
            // By summing we get the total prediction quality.
            NDArray numerator = perExampleLoss.mul(mask).sum();
            // We normalize the loss by the actual number of predictions we had to make
            NDArray denominator = mask.sum().add(1e-5f);
            NDArray result = numerator.div(denominator);

            return scope.ret(result);
        }
    }

    /**
     * Calculates the percentage of correctly predicted masked tokens.
     *
     * @param labels expected tokens and mask
     * @param predictions prediction of a bert model
     * @return the percentage of correctly predicted masked tokens
     */
    public NDArray accuracy(NDList labels, NDList predictions) {
        try (NDManager scope = NDManager.subManagerOf(labels)) {
            scope.tempAttachAll(labels, predictions);

            NDArray mask = labels.get(maskIdx).flatten(); // (B * I)
            NDArray targetIds = labels.get(labelIdx).flatten(); // (B * I)
            NDArray logProbs = predictions.get(logProbsIdx); // (B * I, D)
            NDArray predictedIs = logProbs.argMax(1).toType(DataType.INT32, false); // (B * I)
            NDArray equal = predictedIs.eq(targetIds).mul(mask);
            NDArray equalCount = equal.sum().toType(DataType.FLOAT32, false);
            NDArray count = mask.sum().toType(DataType.FLOAT32, false);
            NDArray result = equalCount.div(count);

            return scope.ret(result);
        }
    }
}
