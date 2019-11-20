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
package ai.djl.ndarray.index;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.types.Shape;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Stream;

/**
 * The {@code NDIndex} allows you to specify a subset of an NDArray that can be used for fetching or
 * updating.
 *
 * <p>It accepts a different index option for each dimension, given in the order of the dimensions.
 * Each dimension has options corresponding to:
 *
 * <ul>
 *   <li>Return all dimensions - Pass null to addIndices
 *   <li>A single value in the dimension - Pass the value to addIndices with a negative index -i
 *       corresponding to [dimensionLength - i]
 *   <li>A range of values - Use addSliceDim
 * </ul>
 *
 * <p>We recommend creating the NDIndex using {@link #NDIndex(String)}.
 *
 * @see #NDIndex(String)
 */
public class NDIndex {

    private static final Pattern ITEM_PATTERN =
            Pattern.compile("(\\*)|((-?\\d+)?:(-?\\d+)?(:(-?\\d+))?)|(-?\\d+)");

    private int rank;
    private List<NDIndexElement> indices;

    /** Creates an empty {@link NDIndex} to append values to. */
    public NDIndex() {
        rank = 0;
        indices = new ArrayList<>();
    }

    /**
     * Creates a {@link NDIndex} given the index values.
     *
     * <p>Here are some examples of the indices format.
     *
     * <pre>
     *     NDArray a = manager.ones(new Shape(5, 4, 3));
     *
     *     // Gets a subsection of the NDArray in the first axis.
     *     assertEquals(a.get(new NDIndex("2")).getShape(), new Shape(4, 3));
     *
     *     // Gets a subsection of the NDArray indexing from the end (-i == length - i).
     *     assertEquals(a.get(new NDIndex("-1")).getShape(), new Shape(4, 3));
     *
     *     // Gets everything in the first axis and a subsection in the second axis.
     *     // You can use either : or * to represent everything
     *     assertEquals(a.get(new NDIndex(":, 2")).getShape(), new Shape(5, 3));
     *     assertEquals(a.get(new NDIndex("*, 2")).getShape(), new Shape(5, 3));
     *
     *     // Gets a range of values along the second axis that is inclusive on the bottom and exclusive on the top.
     *     assertEquals(a.get(new NDIndex(":, 1:3")).getShape(), new Shape(5, 2, 3));
     *
     *     // Excludes either the min or the max of the range to go all the way to the beginning or end.
     *     assertEquals(a.get(new NDIndex(":, :3")).getShape(), new Shape(5, 3, 3));
     *     assertEquals(a.get(new NDIndex(":, 1:")).getShape(), new Shape(5, 4, 3));
     *
     *     // Uses the value after the second colon in a slicing range, the step, to get every other result.
     *     assertEquals(a.get(new NDIndex(":, 1::2")).getShape(), new Shape(5, 2, 3));
     *
     *     // Uses a negative step to reverse along the dimension.
     *     assertEquals(a.get(new NDIndex("-1")).getShape(), new Shape(5, 4, 3));
     * </pre>
     *
     * @param indices a comma separated list of indices corresponding to either subsections,
     *     everything, or slices on a particular dimension
     * @see <a href="https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html">Numpy
     *     Indexing</a>
     */
    public NDIndex(String indices) {
        this();
        addIndices(indices);
    }

    /**
     * Creates an NDIndex with the given indices as specified values on the NDArray.
     *
     * @param indices the indices with each index corresponding to the dimensions and negative
     *     indices starting from the end
     */
    public NDIndex(long... indices) {
        this();
        addIndices(indices);
    }

    /**
     * Returns the number of dimensions specified in the Index.
     *
     * @return the number of dimensions specified in the Index
     */
    public int getRank() {
        return rank;
    }

    /**
     * Returns the index affecting the given dimension.
     *
     * @param dimension the affected dimension
     * @return the index affecting the given dimension
     */
    public NDIndexElement get(int dimension) {
        return indices.get(dimension);
    }

    /**
     * Returns the indices.
     *
     * @return the indices
     */
    public List<NDIndexElement> getIndices() {
        return indices;
    }

    /**
     * Updates the NDIndex by appending indices to the array.
     *
     * @param indices the indices to add similar to {@link #NDIndex(String)}
     * @return the updated {@link NDIndex}
     * @see #NDIndex(String)
     */
    public final NDIndex addIndices(String indices) {
        String[] indexItems = indices.split(",");
        rank += indexItems.length;
        for (String indexItem : indexItems) {
            addIndexItem(indexItem);
        }
        return this;
    }

    /**
     * Updates the NDIndex by appending indices as specified values on the NDArray.
     *
     * @param indices with each index corresponding to the dimensions and negative indices starting
     *     from the end
     * @return the updated {@link NDIndex}
     */
    public final NDIndex addIndices(long... indices) {
        rank += indices.length;
        for (long i : indices) {
            this.indices.add(new NDIndexFixed(i));
        }
        return this;
    }

    /**
     * Updates the NDIndex by appending a boolean NDArray.
     *
     * <p>The NDArray should have a matching shape to the dimensions being fetched and will return
     * where the values in NDIndex do not equal zero.
     *
     * @param index a boolean NDArray where all nonzero elements correspond to elements to return
     * @return the updated {@link NDIndex}
     */
    public NDIndex addBooleanIndex(NDArray index) {
        rank += index.getShape().dimension();
        indices.add(new NDIndexBooleans(index));
        return this;
    }

    /**
     * Appends a new index to slice the dimension and returns a range of values.
     *
     * @param min the minimum of the range
     * @param max the maximum of the range
     * @return the updated {@link NDIndex}
     */
    public NDIndex addSliceDim(long min, long max) {
        rank++;
        indices.add(new NDIndexSlice(min, max, null));
        return this;
    }

    /**
     * Appends a new index to slice the dimension and returns a range of values.
     *
     * @param min the minimum of the range
     * @param max the maximum of the range
     * @param step the step of the slice
     * @return the updated {@link NDIndex}
     */
    public NDIndex addSliceDim(long min, long max, long step) {
        rank++;
        indices.add(new NDIndexSlice(min, max, step));
        return this;
    }

    /**
     * Returns a stream of the NDIndexElements.
     *
     * @return a stream of the NDIndexElements
     */
    public Stream<NDIndexElement> stream() {
        return indices.stream();
    }

    private void addIndexItem(String indexItem) {
        indexItem = indexItem.trim();
        Matcher m = ITEM_PATTERN.matcher(indexItem);
        if (!m.matches()) {
            throw new IllegalArgumentException("Invalid argument index: " + indexItem);
        }

        String star = m.group(1);
        if (star != null) {
            indices.add(new NDIndexAll());
            return;
        }

        String digit = m.group(7);
        if (digit != null) {
            indices.add(new NDIndexFixed(Long.parseLong(digit)));
            return;
        }

        // Slice
        Long min = m.group(3) != null ? Long.parseLong(m.group(3)) : null;
        Long max = m.group(4) != null ? Long.parseLong(m.group(4)) : null;
        Long step = m.group(6) != null ? Long.parseLong(m.group(6)) : null;
        if (min == null && max == null && step == null) {
            indices.add(new NDIndexAll());
        } else {
            indices.add(new NDIndexSlice(min, max, step));
        }
    }

    /**
     * Returns this index as a full slice if it can be represented as one.
     *
     * @param target the shape to index
     * @return the full slice if it can be represented as one
     */
    public Optional<NDIndexFullSlice> getAsFullSlice(Shape target) {
        if (!stream().allMatch(
                        ie ->
                                ie instanceof NDIndexAll
                                        || ie instanceof NDIndexFixed
                                        || ie instanceof NDIndexSlice)) {
            return Optional.empty();
        }
        int indDimensions = getRank();
        int targetDimensions = target.dimension();
        if (indDimensions > target.dimension()) {
            throw new IllegalArgumentException(
                    "The index has too many dimensions - "
                            + indDimensions
                            + " dimensions for array with "
                            + targetDimensions
                            + " dimensions");
        }
        long[] min = new long[targetDimensions];
        long[] max = new long[targetDimensions];
        long[] step = new long[targetDimensions];
        List<Integer> toSqueeze = new ArrayList<>(targetDimensions);
        long[] shape = new long[targetDimensions];
        List<Long> squeezedShape = new ArrayList<>(targetDimensions);
        for (int i = 0; i < indDimensions; i++) {
            NDIndexElement ie = get(i);
            if (ie instanceof NDIndexFixed) {
                min[i] = ((NDIndexFixed) ie).getIndex();
                max[i] = ((NDIndexFixed) ie).getIndex() + 1;
                step[i] = 1;
                toSqueeze.add(i);
                shape[i] = 1;
            } else if (ie instanceof NDIndexSlice) {
                NDIndexSlice slice = (NDIndexSlice) ie;
                min[i] = Optional.ofNullable(slice.getMin()).orElse(0L);
                max[i] = Optional.ofNullable(slice.getMax()).orElse(target.size(i));
                step[i] = Optional.ofNullable(slice.getStep()).orElse(1L);
                if (step[i] > 0) {
                    shape[i] = (max[i] - min[i] - 1) / (step[i] + 1);
                } else {
                    shape[i] = (min[i] - max[i]) / (-step[i] + 1);
                }
                squeezedShape.add(shape[i]);
            } else if (ie instanceof NDIndexAll) {
                min[i] = 0;
                max[i] = target.size(i);
                step[i] = 1;
                shape[i] = target.size(i);
                squeezedShape.add(target.size(i));
            }
        }
        for (int i = indDimensions; i < target.dimension(); i++) {
            min[i] = 0;
            max[i] = target.size(i);
            step[i] = 1;
            shape[i] = target.size(i);
            squeezedShape.add(target.size(i));
        }
        NDIndexFullSlice fullSlice =
                new NDIndexFullSlice(
                        min, max, step, toSqueeze, new Shape(shape), new Shape(squeezedShape));
        return Optional.of(fullSlice);
    }
}
