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

    private static final long serialVersionUID = 1L;

    private List<Integer> arraysToPad;
    private List<Integer> dimsToPad;
    private transient List<NDArraySupplier> paddingSuppliers;
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
            int arrayIndex = arraysToPad.get(i);
            int dimIndex = dimsToPad.get(i);
            NDArray padding = paddingSuppliers.get(i).get(manager);
            long paddingSize = paddingSizes.get(i);
            long maxSize = findMaxSize(inputs, arrayIndex, dimIndex);
            if (paddingSize != -1 && maxSize > paddingSize) {
                throw new IllegalArgumentException(
                        "The batchifier padding size is too small " + maxSize + " " + paddingSize);
            }
            maxSize = Math.max(maxSize, paddingSize);
            long[] arrayValidLengths = padArrays(inputs, arrayIndex, dimIndex, padding, maxSize);
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
     * Finds the maximum size for a particular array/dimension in a batch of inputs (which can be
     * padded to equalize their sizes).
     *
     * @param inputs the batch of inputs
     * @param arrayIndex the array (for each NDList in the batch)
     * @param dimIndex for the array in each NDList in the batch
     * @return the maximum size
     */
    public static long findMaxSize(NDList[] inputs, int arrayIndex, int dimIndex) {
        long maxSize = -1;
        for (NDList input : inputs) {
            NDArray array = input.get(arrayIndex);
            maxSize = Math.max(maxSize, array.getShape().get(dimIndex));
        }
        return maxSize;
    }

    /**
     * Pads the arrays at a particular dimension to all have the same size (updating inputs in
     * place).
     *
     * @param inputs the batch of inputs
     * @param arrayIndex the array (for each NDList in the batch)
     * @param dimIndex for the array in each NDList in the batch
     * @param padding the padding to use. Say you have a batch of arrays of Shape(10, ?, 3) and you
     *     are padding the "?" dimension. There are two padding modes:
     *     <ul>
     *       <li>If you give padding of Shape(1, 3) (same dimensionality as required), it will be
     *           repeated with {@link NDArray#repeat(long)} as necessary
     *       <li>If you give padding of Shape(3) or Shape(0) (smaller dimensionality as required),
     *           it will be broadcasted with {@link NDArray#broadcast(Shape)} to reach the full
     *           required Shape(?, 3)
     *     </ul>
     *
     * @param maxSize the size that each array will be padded to in that dimension. In the example
     *     above, the padding to be applied to the "?" dimension.
     * @return the original valid length for each dimension in the batch (same length as
     *     inputs.length). The inputs will be updated in place.
     */
    public static long[] padArrays(
            NDList[] inputs, int arrayIndex, int dimIndex, NDArray padding, long maxSize) {
        long[] arrayValidLengths = new long[inputs.length];
        for (int i = 0; i < inputs.length; i++) {
            NDArray array = inputs[i].get(arrayIndex);
            String arrayName = array.getName();
            long validLength = array.getShape().get(dimIndex);
            if (validLength < maxSize) {

                // Number of dimensions the padding must be
                int dimensionsRequired =
                        array.getShape().dimension() - padding.getShape().dimension();

                NDArray paddingArray;
                if (dimensionsRequired == 0) {
                    paddingArray =
                            padding.repeat(
                                    Shape.update(
                                            array.getShape(), dimIndex, maxSize - validLength));
                } else if (dimensionsRequired > 0) {
                    paddingArray =
                            padding.broadcast(
                                    Shape.update(
                                            array.getShape(), dimIndex, maxSize - validLength));
                } else {
                    throw new IllegalArgumentException(
                            "The padding must be <="
                                    + dimensionsRequired
                                    + " dimensions, but found "
                                    + padding.getShape().dimension());
                }
                array = array.concat(paddingArray.toType(array.getDataType(), false), dimIndex);
            }
            // keep input name
            array.setName(arrayName);
            inputs[i].set(arrayIndex, array);

            arrayValidLengths[i] = validLength;
        }
        return arrayValidLengths;
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
