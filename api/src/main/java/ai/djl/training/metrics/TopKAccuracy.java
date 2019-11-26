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
package ai.djl.training.metrics;

import ai.djl.ndarray.NDArray;
import java.util.stream.IntStream;

/**
 * {@code TopKAccuracy} is a {@link TrainingMetric} that computes the accuracy of the top k
 * predictions.
 *
 * <p>{@code TopKAccuracy} differs from {@link Accuracy} in that it considers the prediction to be
 * `True` as long as the ground truth label is in the top K predicated labels. If `top_k = 1`, then
 * {@code TopKAccuracy} is identical to {@code Accuracy}.
 */
public class TopKAccuracy extends Accuracy {

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
     * Creates an instance of {@code TopKAccuracy} metric that computes topK accuracy across axis 1
     * along the given index.
     *
     * @param index the index of the {@link NDArray} in labels to compute topK accuracy for
     * @param topK the value of K
     */
    public TopKAccuracy(int index, int topK) {
        this("Top_" + topK + "_Accuracy", index, topK);
    }

    /**
     * Creates an instance of {@code TopKAccuracy} metric that computes topK accuracy across axis 1
     * along the 0th index.
     *
     * @param topK the value of K
     */
    public TopKAccuracy(int topK) {
        this("Top_" + topK + "_Accuracy", 0, topK);
    }

    /** {@inheritDoc} */
    @Override
    public void update(NDArray labels, NDArray predictions) {
        // number of labels and predictions should be the same
        checkLabelShapes(labels, predictions);
        if (predictions.getShape().dimension() > 2) {
            throw new IllegalStateException("Prediction should be less than 2 dimensions");
        }
        // ascending by default
        NDArray topKPrediction = predictions.argSort(axis);
        int numDims = topKPrediction.getShape().dimension();
        if (numDims == 1) {
            addCorrectInstances(
                    topKPrediction.flatten().eq(labels.flatten()).countNonzero().getLong());
        } else if (numDims == 2) {
            int numClasses = (int) topKPrediction.getShape().get(1);
            topK = Math.min(topK, numClasses);
            IntStream.range(0, topK)
                    .forEach(
                            j -> {
                                // get from last index as argSort is ascending
                                NDArray jPrediction =
                                        topKPrediction.get(":, " + (numClasses - j - 1));
                                addCorrectInstances(
                                        jPrediction
                                                .flatten()
                                                .eq(labels.flatten())
                                                .countNonzero()
                                                .getLong());
                            });
        }
        addTotalInstances((int) labels.getShape().get(0));
    }
}
