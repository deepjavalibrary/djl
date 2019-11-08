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
package ai.djl.training.dataset;

import ai.djl.ndarray.NDList;

/**
 * {@code Record} represents a single element of data and labels from {@link Dataset}.
 *
 * <p>The data and labels in record are in the form of an {@link NDList}. This allows it to hold
 * multiple types of data and labels. However, note that the {@link NDList} does not include a
 * dimension for batch.
 */
public class Record {

    private NDList data;
    private NDList labels;

    /**
     * Creates a new instance of {@code Record} with a single element of data and its corresponding
     * labels.
     *
     * @param data an {@link NDList} that contains a single element of data
     * @param labels an {@link NDList} that contains the corresponding label
     */
    public Record(NDList data, NDList labels) {
        this.data = data;
        this.labels = labels;
    }

    /**
     * Gets the data of this {@code Record}.
     *
     * @return an {@link NDList} that contains the data of this {@code Record}
     */
    public NDList getData() {
        return data;
    }

    /**
     * Gets the labels that correspond to the data of this {@code Record}.
     *
     * @return an {@link NDList} that contains label that correspond to the data of this {@code
     *     Record}
     */
    public NDList getLabels() {
        return labels;
    }
}
