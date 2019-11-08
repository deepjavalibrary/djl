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
package ai.djl.translate;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import java.util.Arrays;
import java.util.stream.IntStream;

/**
 * {@code StackBatchifier} is used to merge a list of samples to form a mini-batch of NDArray(s).
 * The is default {@link Batchifier} for data loading.
 */
public class StackBatchifier implements Batchifier {

    /** {@inheritDoc} */
    @Override
    public NDList batchify(NDList[] inputs) {
        // each input as NDList might contain several data or labels
        // so those should be batchified with counterpart
        int size = inputs[0].size();
        // if the NDList is empty
        if (size == 0) {
            return new NDList();
        }
        // collect all the data0...n in batch into one NDList
        NDList[] dataList = new NDList[size];
        for (NDList input : inputs) {
            for (int i = 0; i < size; i++) {
                if (dataList[i] == null) {
                    dataList[i] = new NDList();
                }
                dataList[i].add(input.get(i));
            }
        }

        // stack all the data and labels together
        NDList result = new NDList(size);
        for (NDList list : dataList) {
            NDArray stacked = NDArrays.stack(list);
            result.add(stacked);
            // close the intermediate NDArray
            list.close();
        }

        return result;
    }

    /** {@inheritDoc} */
    @Override
    public NDList[] unbatchify(NDList inputs) {
        if (inputs.size() == 0) {
            return new NDList[0];
        }
        int size = Math.toIntExact(inputs.head().size(0));
        if (size == 0) {
            return new NDList[0];
        }

        NDList[] dataList = new NDList[size];
        for (int i = 0; i < size; i++) {
            dataList[i] = new NDList();
        }

        for (NDArray input : inputs) {
            NDList splitList = input.split(size);
            for (int i = 0; i < size; i++) {
                NDArray array = splitList.get(i).squeeze(0);
                array.setName(input.getName());
                dataList[i].add(array);
            }
        }
        return dataList;
    }

    /** {@inheritDoc} */
    @Override
    public NDList[] split(NDList list, int numOfSlices, boolean evenSplit) {
        int batchSize = Math.toIntExact(list.head().size(0));
        numOfSlices = Math.min(numOfSlices, batchSize);

        NDList[] splitted = new NDList[numOfSlices];
        Arrays.setAll(splitted, i -> new NDList());

        for (NDArray nd : list) {
            String name = nd.getName();
            NDList rows = split(nd, numOfSlices, evenSplit);

            for (int i = 0; i < numOfSlices; ++i) {
                NDArray array = rows.get(i);
                array.setName(name);
                splitted[i].add(array);
            }
        }
        return splitted;
    }

    /**
     * Splits an {@code NDArray} into the given number of slices along the given batch axis.
     *
     * <p>Usually used for data parallelism where each slices is sent to one device (i.e. GPU).
     *
     * @param array a batch of {@code NDArray}.
     * @param numOfSlices number of desired slices.
     * @param evenSplit whether to force all slices to have the same number of elements.
     * @return return value is a NDList even if `numOfSlice` is 1.
     */
    private NDList split(NDArray array, int numOfSlices, boolean evenSplit) {
        int size = Math.toIntExact(array.size(0));
        if (size < numOfSlices) {
            throw new IllegalArgumentException(
                    "Batch size(" + size + ") is less then slice number(" + numOfSlices + ").");
        }

        if (evenSplit && size % numOfSlices != 0) {
            throw new IllegalArgumentException(
                    "data with shape "
                            + size
                            + " cannot be evenly split into "
                            + numOfSlices
                            + ". Use a batch size that's multiple of "
                            + numOfSlices
                            + " or set even_split=true to allow"
                            + " uneven partitioning of data.");
        }

        if (evenSplit) {
            return array.split(numOfSlices);
        }

        int step = (int) Math.ceil((double) size / numOfSlices);
        int[] indices = IntStream.range(1, numOfSlices).map(i -> i * step).toArray();
        return array.split(indices);
    }
}
