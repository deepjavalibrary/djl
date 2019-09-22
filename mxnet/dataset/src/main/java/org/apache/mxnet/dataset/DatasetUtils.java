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
package org.apache.mxnet.dataset;

import java.util.stream.IntStream;
import software.amazon.ai.Device;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDList;

public final class DatasetUtils {
    private DatasetUtils() {}
    /**
     * Splits an {@code NDArray} into `numOfSlice` slices along `batchAxis`.
     *
     * <p>Usually used for data parallelism where each slices is sent to one device (i.e. GPU).
     *
     * @param data a batch of {@code NDArray}.
     * @param numOfSlice number of desired slices.
     * @param batchAxis the axis along which to slice.
     * @param evenSplit whether to force all slices to have the same number of elements.
     * @return return value is a NDList even if `numOfSlice` is 1.
     */
    public static NDList splitData(NDArray data, int numOfSlice, int batchAxis, boolean evenSplit) {
        long size = data.size(batchAxis);
        if (evenSplit && size % numOfSlice != 0) {
            throw new IllegalArgumentException(
                    "data with shape "
                            + size
                            + " cannot be evenly split into "
                            + numOfSlice
                            + " slices along axis "
                            + batchAxis
                            + ". Use a batch size that's multiple of "
                            + numOfSlice
                            + " or set even_split=true to allow"
                            + " uneven partitioning of data.");
        }
        // if size < numOfSlice, decrease numOfSlice to size
        if (!evenSplit && size < numOfSlice) {
            numOfSlice = Math.toIntExact(size);
        }
        long step = size / numOfSlice;
        NDList slices;
        if (evenSplit) {
            slices = data.split(numOfSlice, batchAxis);
        } else {
            int[] indices =
                    IntStream.range(1, numOfSlice).map(i -> Math.toIntExact(i * step)).toArray();
            slices = data.split(indices, batchAxis);
        }
        return slices;
    }
    /**
     * Splits an {@code NDArray} into `devices.length` slices along first axis and loads each slice
     * to one device in `devices`.
     *
     * @param data a batch of {@code NDArray}.
     * @param devices list of {@code Device}.
     * @param evenSplit whether to force all slices to have the same number of elements.
     * @return list of {@code NDArray}, each of whom corresponds to a device in `devices`.
     */
    public static NDList splitAndLoad(NDArray data, Device[] devices, boolean evenSplit) {
        return splitAndLoad(data, devices, 0, evenSplit);
    }

    /**
     * Splits an {@code NDArray} into `devices.length` slices along `batchAxis` and loads each slice
     * to one device in `devices`.
     *
     * @param data a batch of {@code NDArray}.
     * @param devices list of {@code Device}.
     * @param batchAxis the axis along which to slice.
     * @param evenSplit whether to force all slices to have the same number of elements.
     * @return list of {@code NDArray}, each of whom corresponds to a device in `devices`.
     */
    public static NDList splitAndLoad(
            NDArray data, Device[] devices, int batchAxis, boolean evenSplit) {
        // null check
        if (devices == null) {
            throw new IllegalArgumentException("please specify valid devices");
        }
        if (devices.length == 1) {
            return new NDList(data.asInDevice(devices[0], false));
        }
        NDList splices = splitData(data, devices.length, batchAxis, evenSplit);
        return splices.asInDevice(devices, false);
    }
}
