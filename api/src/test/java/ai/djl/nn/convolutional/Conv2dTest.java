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
import java.util.stream.IntStream;
import org.testng.Assert;
import org.testng.annotations.Test;

public class Conv2dTest extends OutputShapeTest {

    @Test
    public void testOutputShapes() {
        Range filters = Range.ofClosed(1, 3);

        Range heights = Range.ofClosed(1, 3);
        Range widths = Range.ofClosed(1, 3);

        Range kernelHeightRange = Range.ofClosed(1, 3);
        Range kernelWidthRange = Range.ofClosed(1, 3);

        Range paddingHeightRange = Range.ofClosed(0, 3);
        Range paddingWidthRange = Range.ofClosed(0, 3);

        Range strideHeightRange = Range.ofClosed(1, 3);
        Range strideWidthRange = Range.ofClosed(1, 3);

        Range dilationHeightRange = Range.ofClosed(1, 3);
        Range dilationWidthRange = Range.ofClosed(1, 3);

        long rows =
                filters.size()
                        * heights.size()
                        * widths.size()
                        * kernelHeightRange.size()
                        * kernelWidthRange.size()
                        * paddingHeightRange.size()
                        * paddingWidthRange.size()
                        * strideHeightRange.size()
                        * strideWidthRange.size()
                        * dilationHeightRange.size()
                        * dilationWidthRange.size();

        IntStream streamToTest;
        if (Boolean.getBoolean("nightly")) {
            // During nightly testing, test all rows
            streamToTest = IntStream.range(0, (int) rows);
        } else {
            // During unit testing, only test the first and last
            streamToTest = IntStream.of(0, (int) rows - 1);
        }

        streamToTest
                .mapToObj(TestData::new)
                .peek(
                        td -> {
                            long ix = td.getIndex();

                            Pair<Long, Long> option = Range.toValue(ix, filters);
                            td.setFilters(option.getKey().intValue() * 128);
                            ix = option.getValue();

                            option = Range.toValue(ix, heights);
                            td.setHeight(option.getKey().intValue() * 128);
                            ix = option.getValue();

                            option = Range.toValue(ix, widths);
                            td.setWidth(option.getKey().intValue() * 128);
                            ix = option.getValue();

                            Pair<Shape, Long> shaped =
                                    Range.toShape(ix, kernelHeightRange, kernelWidthRange);
                            td.setKernel(shaped.getKey());
                            ix = shaped.getValue();

                            shaped = Range.toShape(ix, paddingHeightRange, paddingWidthRange);
                            td.setPadding(shaped.getKey());
                            ix = shaped.getValue();

                            shaped = Range.toShape(ix, strideHeightRange, strideWidthRange);
                            td.setStride(shaped.getKey());
                            ix = shaped.getValue();

                            shaped = Range.toShape(ix, dilationHeightRange, dilationWidthRange);
                            td.setDilation(shaped.getKey());
                        })
                .forEach(this::assertOutputShapes);
    }

    public void assertOutputShapes(TestData data) {
        Shape inputShape = new Shape(1, data.getFilters(), data.getHeight(), data.getWidth());
        long expectedHeight =
                ShapeUtils.convolutionDimensionCalculation(
                        data.getHeight(),
                        data.getKernel().get(0),
                        data.getPadding().get(0),
                        data.getStride().get(0),
                        data.getDilation().get(0));
        long expectedWidth =
                ShapeUtils.convolutionDimensionCalculation(
                        data.getWidth(),
                        data.getKernel().get(1),
                        data.getPadding().get(1),
                        data.getStride().get(1),
                        data.getDilation().get(1));

        Conv2d.Builder builder =
                new Conv2d.Builder()
                        .setFilters(data.getFilters())
                        .setKernelShape(data.getKernel())
                        .optPadding(data.getPadding())
                        .optStride(data.getStride())
                        .optDilation(data.getDilation());

        Shape output = ShapeUtils.outputShapeForBlock(manager, builder.build(), inputShape);
        Assert.assertEquals(output, new Shape(1, data.getFilters(), expectedHeight, expectedWidth));
    }
}
