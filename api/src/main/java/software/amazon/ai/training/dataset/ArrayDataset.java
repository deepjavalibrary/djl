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
package software.amazon.ai.training.dataset;

import java.util.stream.Stream;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.util.Pair;

/**
 * Dataset wrapping {@link NDArray}s. It is able to combine multiple data and labels. Each sample
 * will be retrieved by indexing {@link NDArray}s along the first dimension.
 *
 * <p>The following is an example of how to use ArrayDataset:
 *
 * <pre>
 *     ArrayDataset dataset = new ArrayDataset(
 *                      new NDArray[]{data1, data2},
 *                      new NDArray[]{label1, label2, label3},
 *                      new DataLoadingConfiguration.builder
 *                              .setBatchSize(20)
 *                              .build())
 * </pre>
 *
 * @see Dataset
 */
public final class ArrayDataset extends RandomAccessDataset {

    private final NDArray[] data;
    private final NDArray[] labels;
    private long size;

    public ArrayDataset(NDArray data, DataLoadingConfiguration config) {
        this(new NDArray[] {data}, config);
    }

    public ArrayDataset(NDArray[] data, DataLoadingConfiguration config) {
        this(data, null, config);
    }

    public ArrayDataset(NDArray data, NDArray labels, DataLoadingConfiguration config) {
        this(new NDArray[] {data}, new NDArray[] {labels}, config);
    }

    public ArrayDataset(NDArray[] data, NDArray[] labels, DataLoadingConfiguration config) {
        super(config);
        if (data != null && data.length != 0) {
            size = data[0].size(0);
        } else if (labels != null && labels.length != 0) {
            size = labels[0].size(0);
        } else {
            throw new IllegalArgumentException("Either data or labels must have NDArray");
        }
        // check data and labels have the same size
        if (data != null && Stream.of(data).anyMatch(array -> array.size(0) != size)) {
            throw new IllegalArgumentException("All the NDArray must have the same length!");
        }
        if (labels != null && Stream.of(labels).anyMatch(array -> array.size(0) != size)) {
            throw new IllegalArgumentException("All the NDArray must have the same length!");
        }
        setSize(size);
        this.data = data;
        this.labels = labels;
    }

    @Override
    public Pair<NDList, NDList> get(long index) {
        NDList datum = new NDList();
        NDList label = new NDList();
        for (NDArray array : data) {
            datum.add(array.get(index));
        }
        for (NDArray array : labels) {
            label.add(array.get(index));
        }
        return new Pair<>(datum, label);
    }
}
