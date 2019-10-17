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

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.util.Pair;
import java.util.Arrays;
import java.util.stream.IntStream;

/** Batch is used to get a batch of data and labels from {@link Dataset}. */
public class Batch implements AutoCloseable {

    private NDManager manager;
    private NDList data;
    private NDList labels;

    public Batch(NDManager manager, NDList data, NDList labels) {
        this.manager = manager;
        data.attach(manager);
        labels.attach(manager);
        this.data = data;
        this.labels = labels;
    }

    public NDManager getManager() {
        return manager;
    }

    public NDList getData() {
        return data;
    }

    public NDList getLabels() {
        return labels;
    }

    @Override
    public void close() {
        manager.close();
        manager = null;
    }

    public Batch[] split(Device[] devices, boolean evenSplit) {
        int size = devices.length;
        if (size == 1) {
            NDList d = data.asInDevice(devices[0], true);
            NDList l = labels.asInDevice(devices[0], true);
            return new Batch[] {new Batch(manager, d, l)};
        }

        NDList[] splittedData = split(data, size, evenSplit);
        NDList[] splittedLabels = split(labels, size, evenSplit);

        Batch[] splitted = new Batch[splittedData.length];
        for (int i = 0; i < splittedData.length; ++i) {
            NDList d = splittedData[i].asInDevice(devices[i], true);
            NDList l = splittedLabels[i].asInDevice(devices[i], true);
            splitted[i] = new Batch(manager, d, l);
        }
        return splitted;
    }

    private static NDList[] split(NDList list, int numOfSlice, boolean evenSplit) {
        int batchSize = Math.toIntExact(list.head().size(0));
        numOfSlice = Math.min(numOfSlice, batchSize);

        NDList[] splitted = new NDList[numOfSlice];
        Arrays.setAll(splitted, i -> new NDList());

        for (Pair<String, NDArray> pair : list) {
            String name = pair.getKey();
            NDArray nd = pair.getValue();
            NDList rows = split(nd, numOfSlice, evenSplit);

            for (int i = 0; i < numOfSlice; ++i) {
                splitted[i].add(name, rows.get(i));
            }
        }
        return splitted;
    }

    /**
     * Splits an {@code NDArray} into `numOfSlice` slices along `batchAxis`.
     *
     * <p>Usually used for data parallelism where each slices is sent to one device (i.e. GPU).
     *
     * @param array a batch of {@code NDArray}.
     * @param numOfSlice number of desired slices.
     * @param evenSplit whether to force all slices to have the same number of elements.
     * @return return value is a NDList even if `numOfSlice` is 1.
     */
    private static NDList split(NDArray array, int numOfSlice, boolean evenSplit) {
        int size = Math.toIntExact(array.size(0));
        if (size < numOfSlice) {
            throw new IllegalArgumentException(
                    "Batch size(" + size + ") is less then slice number(" + numOfSlice + ").");
        }

        if (evenSplit && size % numOfSlice != 0) {
            throw new IllegalArgumentException(
                    "data with shape "
                            + size
                            + " cannot be evenly split into "
                            + numOfSlice
                            + ". Use a batch size that's multiple of "
                            + numOfSlice
                            + " or set even_split=true to allow"
                            + " uneven partitioning of data.");
        }

        if (evenSplit) {
            return array.split(numOfSlice);
        }

        int step = size / numOfSlice;
        int[] indices = IntStream.range(1, numOfSlice).map(i -> i * step).toArray();
        return array.split(indices);
    }
}
