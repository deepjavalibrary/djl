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
package ai.djl.mxnet.dataset;

import java.util.Arrays;
import java.util.stream.IntStream;
import software.amazon.ai.Device;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.training.dataset.Batch;
import software.amazon.ai.util.Pair;

public final class DatasetUtils {

    private DatasetUtils() {}

    public static Batch[] split(Batch batch, Device[] devices, boolean evenSplit) {
        int size = devices.length;
        if (size == 1) {
            Batch[] splitted = new Batch[1];
            NDList data = batch.getData().asInDevice(devices[0], true);
            NDList labels = batch.getLabels().asInDevice(devices[0], true);
            splitted[0] = new Batch(data, labels);
            return splitted;
        }

        NDList[] splittedData = split(batch.getData(), size, evenSplit);
        NDList[] splittedLabels = split(batch.getLabels(), size, evenSplit);

        Batch[] splitted = new Batch[splittedData.length];
        for (int i = 0; i < splittedData.length; ++i) {
            NDList data = splittedData[i].asInDevice(devices[i], true);
            NDList labels = splittedLabels[i].asInDevice(devices[i], true);
            splitted[i] = new Batch(data, labels);
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
