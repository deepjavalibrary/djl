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
package software.amazon.ai.ndarray.index;

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import software.amazon.ai.ndarray.NDArray;

/**
 * The {@code NDIndex} allows you to specify a subset of an NDArray that can be used for fetching or
 * updating.
 *
 * <p>It accepts different index option for each dimension, given in the order of the dimensions.
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
            Pattern.compile("(\\*)|((-?\\d+)?:(-?\\d+)?(:-?\\d+)?)|(-?\\d+)");

    private int rank;
    private List<NDIndexElement> indices;

    /** Creates an empty {@link NDIndex} to append values to. */
    public NDIndex() {}

    /**
     * Creates a {@link NDIndex} given the index values.
     *
     * <p>Here are some examples of the indices format.
     *
     * <pre>
     *     NDArray a = factory.ones(new DataDesc(new Shape(5, 4, 3)));
     *
     *     // Get a subsection of the NDArray in the first axis
     *     assertEquals(a.get(new NDIndex("2")).getShape(), new Shape(4, 3));
     *
     *     // Get a subsection of the NDArray indexing from the end (-i == length - i)
     *     assertEquals(a.get(new NDIndex("-1")).getShape(), new Shape(4, 3));
     *
     *     // Get everything in the first axis and a subsection in the second axis.
     *     // You can use either : or * to represent everything
     *     assertEquals(a.get(new NDIndex(":, 2")).getShape(), new Shape(5, 3));
     *     assertEquals(a.get(new NDIndex("*, 2")).getShape(), new Shape(5, 3));
     *
     *     // Gets a range of values along the second axis that is inclusive on the bottom and exclusive on the top
     *     assertEquals(a.get(new NDIndex(":, 1:3")).getShape(), new Shape(5, 2, 3));
     *
     *     // You can exclude either the min or the max of the range to go all the way to the beginning or end.
     *     assertEquals(a.get(new NDIndex(":, :3")).getShape(), new Shape(5, 3, 3));
     *     assertEquals(a.get(new NDIndex(":, 1:")).getShape(), new Shape(5, 4, 3));
     *
     *     // The value after the second colon in a slicing range is the step. You can use it to get every other result:
     *     assertEquals(a.get(new NDIndex(":, 1::2")).getShape(), new Shape(5, 2, 3));
     *
     *     // Use a negative step to reverse along the dimension
     *     assertEquals(a.get(new NDIndex("-1")).getShape(), new Shape(5, 4, 3));
     * </pre>
     *
     * @param indices A comma separated list of indices corresponding to either subsections,
     *     everything, or slices on a particular dimension
     * @see <a href="https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html">Numpy
     *     Indexing</a>
     */
    public NDIndex(String indices) {
        rank = 0;
        this.indices = new ArrayList<>();
        addIndices(indices);
    }

    /**
     * Creates an NDIndex with the given indices as specified values on the NDArray.
     *
     * @param indices indices with each index corresponding to the dimensions and negative indices
     *     tarting from the end
     */
    public NDIndex(int... indices) {
        rank = 0;
        this.indices = new ArrayList<>(indices.length);
        addIndices(indices);
    }

    /**
     * Returns the number of dimensions specified in the Index.
     *
     * @return Returns the number of dimensions specified in the Index.
     */
    public int getRank() {
        return rank;
    }

    /**
     * Returns the index affecting the given dimension.
     *
     * @param dimension The affected dimension
     * @return Returns the index affecting the given dimension
     */
    public NDIndexElement get(int dimension) {
        return indices.get(dimension);
    }

    /**
     * Updates the NDIndex by appending indices to the array.
     *
     * @param indices the indices to add similar to {@link #NDIndex(String)}
     * @return The updated {@link NDIndex}
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
     * @param indices with each index corresponding to the dimensions and negative indices tarting
     *     from the end
     * @return Returns the updated {@link NDIndex}
     */
    public final NDIndex addIndices(int... indices) {
        rank += indices.length;
        for (Integer i : indices) {
            this.indices.add(new NDIndexFixed(i));
        }
        return this;
    }

    /**
     * Updates the NDIndex by appending a boolean NDArray.
     *
     * <p>The NDArray should have a matching shape to the dimensions being fetched and will return
     * where the values in NDIndex do not equal zero
     *
     * @param index a boolean NDArray where all nonzero elements correspond to elements to return
     * @return Returns the updated {@link NDIndex}
     */
    public NDIndex addBooleanIndex(NDArray index) {
        rank += index.getShape().dimension();
        indices.add(new NDIndexBooleans(index));
        return this;
    }

    /**
     * Appends a new index to slice the dimension and return a range of values.
     *
     * @param min The minimum of the range
     * @param max The maximum of the range
     * @return Returns the updated {@link NDIndex}
     */
    public NDIndex addSliceDim(int min, int max) {
        rank++;
        indices.add(new NDIndexSlice(min, max, null));
        return this;
    }

    /**
     * Appends a new index to slice the dimension and return a range of values.
     *
     * @param min The minimum of the range
     * @param max The maximum of the range
     * @param step The step of the slice
     * @return Returns the updated {@link NDIndex}
     */
    public NDIndex addSliceDim(int min, int max, int step) {
        rank++;
        indices.add(new NDIndexSlice(min, max, step));
        return this;
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

        String digit = m.group(6);
        if (digit != null) {
            indices.add(new NDIndexFixed(Integer.parseInt(digit)));
            return;
        }

        // Slice
        Integer min = m.group(3) != null ? Integer.parseInt(m.group(3)) : null;
        Integer max = m.group(4) != null ? Integer.parseInt(m.group(4)) : null;
        Integer step = m.group(5) != null ? Integer.parseInt(m.group(5)) : null;
        if (min == null && max == null && step == null) {
            indices.add(new NDIndexAll());
        } else {
            indices.add(new NDIndexSlice(min, max, step));
        }
    }
}
