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

import ai.djl.modality.cv.MultiBoxTarget;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import org.testng.Assert;
import org.testng.annotations.Test;

public class MultiBoxTargetTest {
    @Test
    public void testTargets() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray anchorBoxes = manager.ones(new Shape(1, 4096, 4));
            NDArray label = manager.zeros(new Shape(1, 1, 5));
            NDArray classPreds = manager.ones(new Shape(1, 2, 4096));

            MultiBoxTarget multiBoxTarget = new MultiBoxTarget.Builder().build();
            NDList targets = multiBoxTarget.target(new NDList(anchorBoxes, label, classPreds));

            Assert.assertEquals(new Shape(1, 16384), targets.get(0).getShape());
            Assert.assertEquals(new Shape(1, 16384), targets.get(1).getShape());
            Assert.assertEquals(new Shape(1, 4096), targets.get(2).getShape());
        }
    }
}
