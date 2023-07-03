/*
 * Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.modality.cv.output;

import org.testng.Assert;
import org.testng.annotations.Test;

public class RectangleTest {

    @Test
    public void testIoU() {
        BoundingBox box = new Rectangle(1, 3, 4, 5);
        Rectangle rect = new Rectangle(1, 2, 3, 4);
        double iou = box.getIoU(rect);
        Assert.assertEquals(iou, 0.47058823529411764, 0.00001);

        rect = new Rectangle(6, 2, 3, 4);
        iou = box.getIoU(rect);
        Assert.assertEquals(iou, 0);
    }
}
