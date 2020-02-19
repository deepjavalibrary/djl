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

package ai.djl.ndarray.types;

import org.testng.Assert;
import org.testng.annotations.Test;

public class ShapeTest {

    @Test
    public void shapeTest() {
        Shape shape = new Shape(1, 3, 224, 224);
        Assert.assertEquals(shape.getShape(), new long[] {1, 3, 224, 224});
        Assert.assertThrows(IllegalArgumentException.class, () -> shape.size(-1));
        Assert.assertEquals(shape.size(0, 1), 3);
        Assert.assertEquals(shape.size(), 150528);
        Assert.assertEquals(shape.dimension(), 4);
        Assert.assertEquals(shape.slice(1), new Shape(3, 224, 224));
        Assert.assertEquals(shape.head(), 1);
        Assert.assertEquals(shape.toString(), "(1, 3, 224, 224)");
        Assert.assertEquals(shape.getTrailingOnes(), 0);
        Assert.assertEquals(shape.getLeadingOnes(), 1);
    }

    @Test
    public void testHasLayout() {
        Shape withLayout =
                new Shape(
                        new long[] {2, 2}, new LayoutType[] {LayoutType.BATCH, LayoutType.CHANNEL});
        Shape withoutLayout = new Shape(2, 2);

        Assert.assertTrue(withLayout.isLayoutKnown());
        Assert.assertFalse(withoutLayout.isLayoutKnown());
    }
}
