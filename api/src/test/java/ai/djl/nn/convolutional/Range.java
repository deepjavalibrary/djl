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

package ai.djl.nn.convolutional;

import ai.djl.ndarray.types.Shape;
import ai.djl.util.Pair;

/**
 * Utility class for capturing a range of values to be used in a combinatorial spread of values in
 * testing.
 */
public final class Range {

    private final long start;

    private final long end;

    private final boolean closed;

    private Range(long start, long end, boolean closed) {
        this.start = start;
        this.end = end;
        this.closed = closed;
    }

    /**
     * Create an open range of values.
     *
     * @param start the start of a range of value (inclusive)
     * @param end the start of a range of value (exclusive)
     * @return the new {@link Range} object
     */
    public static Range of(long start, long end) {
        return new Range(start, end, false);
    }

    /**
     * Create an closed range of values.
     *
     * @param start the start of a range of value (inclusive)
     * @param end the end of a range of value (inclusive)
     * @return the new {@link Range} object
     */
    public static Range ofClosed(long start, long end) {
        return new Range(start, end, true);
    }

    /**
     * Create a 1D shape with a value between the provided {@link Range}.
     *
     * @param index the combinatorial index used to calculate the value within the {@link Range}
     * @param widthRange the range of values to use for the 1D {@link Shape}
     * @return the calculated {@link Shape} and the new combinatorial index
     */
    public static Pair<Shape, Long> toShape(long index, Range widthRange) {
        long width = widthRange.value(index);
        return new Pair<>(new Shape(width), index / widthRange.size());
    }

    /**
     * Create a 2D shape with a value between the provided {@link Range}s.
     *
     * @param index the combinatorial index used to calculate the value within the {@link Range}
     * @param heightRange the range of values to use for the height component of the 2D {@link
     *     Shape}
     * @param widthRange the range of values to use for the width component of the 2D {@link Shape}
     * @return the calculated {@link Shape} and the new combinatorial index
     */
    public static Pair<Shape, Long> toShape(long index, Range heightRange, Range widthRange) {
        long height = heightRange.value(index);
        index /= heightRange.size();
        long width = widthRange.value(index);
        return new Pair<>(new Shape(height, width), index / widthRange.size());
    }

    /**
     * Create a 3D shape with a value between the provided {@link Range}s.
     *
     * @param index the combinatorial index used to calculate the value within the {@link Range}
     * @param depthRange the range of values to use for the depth component of the 3D {@link Shape}
     * @param heightRange the range of values to use for the height component of the 3D {@link
     *     Shape}
     * @param widthRange the range of values to use for the width component of the 3D {@link Shape}
     * @return the calculated {@link Shape} and the new combinatorial index
     */
    public static Pair<Shape, Long> toShape(
            long index, Range depthRange, Range heightRange, Range widthRange) {
        long depth = depthRange.value(index);
        index /= depthRange.size();
        long height = heightRange.value(index);
        index /= heightRange.size();
        long width = widthRange.value(index);
        return new Pair<>(new Shape(depth, height, width), index / widthRange.size());
    }

    /**
     * Calculate a value between the provided {@link Range}.
     *
     * @param index the combinatorial index used to calculate the value within the {@link Range}
     * @param range the range of values to use for the calculation
     * @return the calculated value and the new combinatorial index
     */
    public static Pair<Long, Long> toValue(long index, Range range) {
        return new Pair<>(range.value(index), index / range.size());
    }

    /**
     * The number of values between the start and end values of this {@link Range}.
     *
     * @return the size of this range
     */
    public long size() {
        if (closed) {
            return end - start + 1;
        } else {
            return end - start;
        }
    }

    /**
     * Calculate a value within the start and end points of this range.
     *
     * @param forIndex the combinatorial index used to calculate the value within the {@link Range}.
     * @return the calculated value
     */
    public long value(long forIndex) {
        return (forIndex % size()) + start;
    }
}
