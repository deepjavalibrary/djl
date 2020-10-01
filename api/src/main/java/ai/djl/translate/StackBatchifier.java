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

import ai.djl.engine.EngineException;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import java.util.Arrays;
import java.util.stream.LongStream;

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
        int batchSize = inputs.length;
        int numInputKinds = inputs[0].size();
        // if the NDList is empty
        if (numInputKinds == 0) {
            return new NDList();
        }

        try {
            // stack all the data and labels together
            NDList result = new NDList(numInputKinds);
            for (int i = 0; i < numInputKinds; i++) {
                NDList inputsOfKind = new NDList(batchSize);
                String inputName = inputs[0].get(i).getName();
                for (NDList input : inputs) {
                    inputsOfKind.add(input.get(i));
                }
                NDArray stacked = NDArrays.stack(new NDList(inputsOfKind));
                // keep the name for stacked inputs
                stacked.setName(inputName);
                result.add(stacked);
            }

            return result;
        } catch (IndexOutOfBoundsException | EngineException e) {
            // If there is an error when batchifying, check for various potential problems with the
            // inputs. The error is not handled in this block. It only attempts to clarify the
            // error's cause before rethrowing.

            // Check if numInputKinds is not consistant for all inputs
            for (NDList input : inputs) {
                if (input.size() != numInputKinds) {
                    throw new IllegalArgumentException(
                            "You cannot batch data with different numbers of inputs", e);
                }
            }

            // Check if data does not have a consistent shape or type
            for (int i = 0; i < numInputKinds; i++) {
                Shape kindDataShape = inputs[0].get(i).getShape();
                DataType kindDataType = inputs[0].get(i).getDataType();
                for (NDList input : inputs) {
                    NDArray currInput = input.get(i);
                    if (!currInput.getShape().equals(kindDataShape)) {
                        throw new IllegalArgumentException(
                                "You cannot batch data with different input shapes", e);
                    }
                    if (!currInput.getDataType().equals(kindDataType)) {
                        throw new IllegalArgumentException(
                                "You cannot batch data with different input data types", e);
                    }
                }
            }

            // Could not clarify cause - rethrow original error.
            throw e;
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDList[] unbatchify(NDList inputs) {
        int numInputKinds = inputs.size();
        if (numInputKinds == 0) {
            return new NDList[0];
        }
        int batchSize = Math.toIntExact(inputs.head().size(0));
        if (batchSize == 0) {
            return new NDList[0];
        }

        NDList[] dataList = new NDList[batchSize];
        for (int i = 0; i < batchSize; i++) {
            dataList[i] = new NDList();
        }

        for (NDArray input : inputs) {
            NDList splitList = input.split(batchSize);
            for (int i = 0; i < batchSize; i++) {
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
     * <p>Usually used for data parallelism where each slice is sent to one device (i.e. GPU).
     *
     * @param array a batch of {@code NDArray}
     * @param numOfSlices the number of desired slices
     * @param evenSplit whether to force all slices to have the same number of elements
     * @return an NDList even if `numOfSlice` is 1
     */
    private NDList split(NDArray array, int numOfSlices, boolean evenSplit) {
        int batchSize = Math.toIntExact(array.size(0));
        if (batchSize < numOfSlices) {
            throw new IllegalArgumentException(
                    "Batch size("
                            + batchSize
                            + ") is less then slice number("
                            + numOfSlices
                            + ").");
        }

        if (evenSplit && batchSize % numOfSlices != 0) {
            throw new IllegalArgumentException(
                    "data with shape "
                            + batchSize
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

        int step = (int) Math.ceil((double) batchSize / numOfSlices);
        long[] indices = LongStream.range(1, numOfSlices).map(i -> i * step).toArray();
        return array.split(indices);
    }
}
