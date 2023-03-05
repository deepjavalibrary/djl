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

import ai.djl.ndarray.NDList;

import java.io.Serializable;
import java.util.Arrays;
import java.util.stream.IntStream;

/**
 * An interface for different strategies to convert between {@link ai.djl.training.dataset.Record}
 * {@link NDList}s and {@link ai.djl.training.dataset.Batch} {@link NDList}.
 *
 * <p>Different implementations of {@code Batchifier} represent different ways of creating batches.
 * The most common would be the {@link StackBatchifier} that batchifies by creating a new batch axis
 * as axis 0. Another implementation could be a concatenated batchifier for sequence elements that
 * will concatenate the data elements along the time axis.
 */
public interface Batchifier extends Serializable {

    Batchifier STACK = new StackBatchifier();

    /**
     * Returns a batchifier from a batchifier name.
     *
     * @param name the batchifier name
     * @return the batchifier with the given name
     * @throws IllegalArgumentException if an invalid name is given
     */
    static Batchifier fromString(String name) {
        switch (name) {
            case "stack":
                return STACK;
            case "padding":
                return new SimplePaddingStackBatchifier();
            case "none":
                return null;
            default:
                throw new IllegalArgumentException("Invalid batchifier name");
        }
    }

    /**
     * Converts an array of {@link ai.djl.training.dataset.Record} {@link NDList}s into a combined
     * {@link ai.djl.training.dataset.Batch} NDList.
     *
     * <p>The size of the input array is the batch size. The data in each of the {@link NDList} are
     * assumed to be the same, and are batched together to form one batched {@link NDList}.
     *
     * @param inputs the input array of {@link NDList} where each element is a
     * @return the batchified {@link NDList}
     */
    NDList batchify(NDList[] inputs);

    /**
     * Splits a combined {@link ai.djl.training.dataset.Batch} {@link NDList} into it's constituent
     * {@link ai.djl.training.dataset.Record} {@link NDList}s.
     *
     * <p>This reverses the {@link #batchify(NDList[]) batchify} operation.
     *
     * @param inputs the {@link NDList} that needs to be 'unbatchified'
     * @return an array of NDLists, of size equal to batch size, where each NDList is one element
     *     from the batch of inputs
     */
    NDList[] unbatchify(NDList inputs);

    /**
     * Partitions the given {@link ai.djl.training.dataset.Batch} {@link NDList} into multiple
     * {@link ai.djl.training.dataset.Batch} lists with smaller batch size.
     *
     * <p>As an example, this function might be used for multi-GPU training where it takes the main
     * batch and splits it into sub-batches that can be run on each GPU.
     *
     * <p>This function unbatchifies the input {@link NDList}, redistributes them into the given
     * number of slices, and then batchify each of the slices to form an array of {@link NDList}.
     *
     * @param list the {@link NDList} that needs to be split
     * @param numOfSlices the number of slices the list must be sliced into
     * @param evenSplit whether each slice must have the same shape
     * @return an array of {@link NDList} that contains all the slices
     */
    default NDList[] split(NDList list, int numOfSlices, boolean evenSplit) {
        NDList[] unbatched = unbatchify(list);
        int batchSize = unbatched.length;
        numOfSlices = Math.min(numOfSlices, batchSize);
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

        NDList[] splitted = new NDList[numOfSlices];
        Arrays.setAll(splitted, i -> new NDList());

        int step = (int) Math.ceil((double) batchSize / numOfSlices);
        for (int i = 0; i < numOfSlices; i++) {
            NDList[] currentUnbatched =
                    IntStream.range(i * step, Math.min((i + 1) * step, batchSize))
                            .mapToObj(j -> unbatched[j])
                            .toArray(NDList[]::new);
            splitted[i] = batchify(currentUnbatched);
        }
        return splitted;
    }
}
