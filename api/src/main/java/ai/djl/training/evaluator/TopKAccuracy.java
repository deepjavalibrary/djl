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
package ai.djl.training.evaluator;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.DataType;
import ai.djl.util.Pair;
import java.util.stream.IntStream;

/**
 * {@code TopKAccuracy} is an {@link Evaluator} that computes the accuracy of the top k predictions.
 *
 * <p>{@code TopKAccuracy} differs from {@link AbstractAccuracy} in that it considers the prediction
 * to be `True` as long as the ground truth label is in the top K predicated labels. If `top_k = 1`,
 * then {@code TopKAccuracy} is identical to {@code Accuracy}.
 */
public class TopKAccuracy extends AbstractAccuracy {

    private int topK;

    /**
     * Creates a {@code TopKAccuracy} instance.
     *
     * @param name the accuracy name, default "Top_K_Accuracy"
     * @param index the index of the {@link NDArray} in labels to compute topK accuracy for
     * @param topK the value of K
     */
    public TopKAccuracy(String name, int index, int topK) {
        super(name, index);
        if (topK > 1) {
            this.topK = topK;
        } else {
            throw new IllegalArgumentException("Please use TopKAccuracy with topK more than 1");
        }
    }

    /**
     * Creates an instance of {@code TopKAccuracy} evaluator that computes topK accuracy across axis
     * 1 along the given index.
     *
     * @param index the index of the {@link NDArray} in labels to compute topK accuracy for
     * @param topK the value of K
     */
    public TopKAccuracy(int index, int topK) {
        this("Top_" + topK + "_Accuracy", index, topK);
    }

    /**
     * Creates an instance of {@code TopKAccuracy} evaluator that computes topK accuracy across axis
     * 1 along the 0th index.
     *
     * @param topK the value of K
     */
    public TopKAccuracy(int topK) {
        this("Top_" + topK + "_Accuracy", 0, topK);
    }

    /** {@inheritDoc} */
    @Override
    protected Pair<Long, NDArray> accuracyHelper(NDList labels, NDList predictions) {
        NDArray label = labels.get(index);
        NDArray prediction = predictions.get(index);
        // number of labels and predictions should be the same
        checkLabelShapes(label, prediction);
        // ascending by default
        NDArray topKPrediction = prediction.argSort(axis).toType(DataType.INT64, false);
        int numDims = topKPrediction.getShape().dimension();
        NDArray numCorrect;
        if (numDims == 1) {
            numCorrect = topKPrediction.flatten().eq(label.flatten()).countNonzero();
        } else if (numDims == 2) {
            int numClasses = (int) topKPrediction.getShape().get(1);
            topK = Math.min(topK, numClasses);
            numCorrect =
                    NDArrays.add(
                            IntStream.range(0, topK)
                                    .mapToObj(
                                            j -> {
                                                // get from last index as argSort is ascending
                                                NDArray jPrediction =
                                                        topKPrediction.get(
                                                                ":, {}", numClasses - j - 1);
                                                return jPrediction
                                                        .flatten()
                                                        .eq(
                                                                label.flatten()
                                                                        .toType(
                                                                                DataType.INT64,
                                                                                false))
                                                        .countNonzero();
                                            })
                                    .toArray(NDArray[]::new));
        } else {
            throw new IllegalArgumentException("Prediction should be less than 2 dimensions");
        }
        long total = label.getShape().get(0);
        return new Pair<>(total, numCorrect);
    }
}
