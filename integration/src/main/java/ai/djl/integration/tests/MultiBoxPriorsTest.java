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
package ai.djl.integration.tests;

import ai.djl.modality.cv.MultiBoxPriors;
import ai.djl.modality.cv.Point;
import ai.djl.modality.cv.Rectangle;
import java.util.Arrays;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.Test;

public class MultiBoxPriorsTest {
    @Test
    public void testMultiBoxPriors() {
        List<Float> sizes = Arrays.asList(0.2f, 0.272f);
        List<Float> ratios = Arrays.asList(1f, 2f, 0.5f);
        MultiBoxPriors multiBoxPriors = new MultiBoxPriors(sizes, ratios, 512, 32, 32);
        List<Rectangle> anchorBoxes = multiBoxPriors.multiBoxPriors(new Point(0, 0));
        Assert.assertEquals(anchorBoxes.size(), 4);
    }

    @Test
    public void testGenerateAnchorBoxes() {
        List<Float> sizes = Arrays.asList(0.2f, 0.272f);
        List<Float> ratios = Arrays.asList(1f, 2f, 0.5f);
        MultiBoxPriors multiBoxPriors = new MultiBoxPriors(sizes, ratios, 512, 32, 32);
        List<Rectangle> anchorBoxes = multiBoxPriors.generateAnchorBoxes();
        Assert.assertEquals(anchorBoxes.size(), 4096);
    }
}
