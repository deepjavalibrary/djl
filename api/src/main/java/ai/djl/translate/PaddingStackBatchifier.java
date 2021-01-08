/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.Shape;
import java.util.ArrayList;
import java.util.List;

/**
 * The padding stack batchifier is a {@link StackBatchifier} that also pads elements to reach the
 * same length.
 */
public final class PaddingStackBatchifier implements Batchifier {

    private List<Integer> arraysToPad;
    private List<Integer> dimsToPad;
    private List<NDArraySupplier> paddingSuppliers;
    private List<Integer> paddingSizes;
    private boolean includeValidLengths;

    private PaddingStackBatchifier(Builder builder) {
        arraysToPad = builder.arraysToPad;
        dimsToPad = builder.dimsToPad;
        paddingSuppliers = builder.paddingSuppliers;
        paddingSizes = builder.paddingSizes;
        includeValidLengths = builder.includeValidLengths;
    }

    /** {@inheritDoc} */
    @Override
    public NDList batchify(NDList[] inputs) {
        NDList validLengths = new NDList(inputs.length);
        NDManager manager = inputs[0].get(0).getManager();
        for (int i = 0; i < arraysToPad.size(); i++) {
            long[] arrayValidLengths = new long[inputs.length];
            int arrayIndex = arraysToPad.get(i);
            int dimIndex = dimsToPad.get(i);
            NDArray padding = paddingSuppliers.get(i).get(manager);
            long paddingSize = paddingSizes.get(i);
            long maxSize = -1;
            for (NDList input : inputs) {
                NDArray array = input.get(arrayIndex);
                maxSize = Math.max(maxSize, array.getShape().get(dimIndex));
            }
            if (paddingSize != -1 && maxSize > paddingSize) {
                throw new IllegalArgumentException(
                        "The batchifier padding size is too small " + maxSize + " " + paddingSize);
            }
            maxSize = Math.max(maxSize, paddingSize);
            for (int j = 0; j < inputs.length; j++) {
                NDArray array = inputs[j].get(arrayIndex);
                String arrayName = array.getName();
                long validLength = array.getShape().get(dimIndex);
                if (validLength < maxSize) {
                    NDArray paddingArray =
                            padding.repeat(
                                    Shape.update(
                                            array.getShape(), dimIndex, maxSize - validLength));
                    array = array.concat(paddingArray.toType(array.getDataType(), false), dimIndex);
                }
                arrayValidLengths[j] = validLength;
                // keep input name
                array.setName(arrayName);
                inputs[j].set(arrayIndex, array);
            }
            validLengths.add(manager.create(arrayValidLengths));
        }
        NDList result = Batchifier.STACK.batchify(inputs);
        if (includeValidLengths) {
            result.addAll(validLengths);
        }
        return result;
    }

    /** {@inheritDoc} */
    @Override
    public NDList[] unbatchify(NDList inputs) {
        if (!includeValidLengths) {
            return Batchifier.STACK.unbatchify(inputs);
        }
        NDList validLengths =
                new NDList(inputs.subList(inputs.size() - arraysToPad.size(), inputs.size()));
        inputs = new NDList(inputs.subList(0, inputs.size() - arraysToPad.size()));
        NDList[] split = Batchifier.STACK.unbatchify(inputs);
        for (int i = 0; i < split.length; i++) {
            NDList arrays = split[i];
            for (int j = 0; j < arraysToPad.size(); j++) {
                long validLength = validLengths.get(j).getLong(i);
                int arrayIndex = arraysToPad.get(j);
                NDArray dePadded =
                        arrays.get(arrayIndex)
                                .get(NDIndex.sliceAxis(dimsToPad.get(j) - 1, 0, validLength));
                arrays.set(arrayIndex, dePadded);
            }
        }
        return split;
    }

    /** {@inheritDoc} */
    @Override
    public NDList[] split(NDList list, int numOfSlices, boolean evenSplit) {
        if (!includeValidLengths) {
            return Batchifier.STACK.split(list, numOfSlices, evenSplit);
        }
        NDList validLengths =
                new NDList(list.subList(list.size() - arraysToPad.size(), list.size()));
        list = new NDList(list.subList(0, list.size() - arraysToPad.size()));
        NDList[] split = Batchifier.STACK.split(list, numOfSlices, evenSplit);
        long sliceSize = split[0].get(0).getShape().get(0);
        long totalSize = list.get(0).getShape().get(0);
        for (int i = 0; i < split.length; i++) {
            // TODO: The padding required may not be the same for all splits. For smaller splits,
            // we can remove some extra padding.
            NDList arrays = split[i];
            for (int j = 0; j < arraysToPad.size(); j++) {
                long min = i * sliceSize;
                long max = Math.min((i + 1) * sliceSize, totalSize);
                NDArray splitValidLenghts = validLengths.get(j).get(NDIndex.sliceAxis(0, min, max));
                arrays.add(splitValidLenghts);
            }
        }
        return split;
    }

    /**
     * Returns a {@link PaddingStackBatchifier.Builder}.
     *
     * @return a {@link PaddingStackBatchifier.Builder}
     */
    public static PaddingStackBatchifier.Builder builder() {
        return new Builder();
    }

    /** Builder to build a {@link PaddingStackBatchifier}. */
    public static final class Builder {

        private List<Integer> arraysToPad;
        private List<Integer> dimsToPad;
        private List<NDArraySupplier> paddingSuppliers;
        private List<Integer> paddingSizes;
        private boolean includeValidLengths;

        private Builder() {
            arraysToPad = new ArrayList<>();
            dimsToPad = new ArrayList<>();
            paddingSuppliers = new ArrayList<>();
            paddingSizes = new ArrayList<>();
        }

        /**
         * Sets whether to include the valid lengths (length of non-padded data) for each array.
         *
         * @param includeValidLengths true to include valid lengths
         * @return this builder
         */
        public Builder optIncludeValidLengths(boolean includeValidLengths) {
            this.includeValidLengths = includeValidLengths;
            return this;
        }

        /**
         * Adds a new dimension to be padded in the input {@link NDList}.
         *
         * @param array which array in the {@link NDList} to pad
         * @param dim which dimension in the array to pad
         * @param supplier a supplier that produces the padding array. The padding array shape
         *     should include both the batch and a 1 for the padded dimension. For batch array shape
         *     NTC, the padding shape should be N x 1 x C
         * @return this builder
         */
        public Builder addPad(int array, int dim, NDArraySupplier supplier) {
            return addPad(array, dim, supplier, -1);
        }

        /**
         * Adds a new dimension to be padded in the input {@link NDList}.
         *
         * @param array which array in the {@link NDList} to pad
         * @param dim which dimension in the array to pad
         * @param supplier a supplier that produces the padding array. The padding array shape
         *     should include both the batch and a 1 for the padded dimension. For batch array shape
         *     NTC, the padding shape should be N x 1 x C
         * @param paddingSize the minimum padding size to use. All sequences to pad must be less
         *     than this size
         * @return this builder
         */
        public Builder addPad(int array, int dim, NDArraySupplier supplier, int paddingSize) {
            arraysToPad.add(array);
            dimsToPad.add(dim);
            paddingSuppliers.add(supplier);
            paddingSizes.add(paddingSize);
            return this;
        }

        /**
         * Builds the {@link PaddingStackBatchifier}.
         *
         * @return the constructed {@link PaddingStackBatchifier}
         */
        public PaddingStackBatchifier build() {
            return new PaddingStackBatchifier(this);
        }
    }
}
