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
package ai.djl.integration.tests.modality.cv;

import ai.djl.integration.util.Assertions;
import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.Shape;
import org.testng.Assert;
import org.testng.annotations.Test;

public class NDImageUtilsTest {

    @Test
    public void testNormalize() {
        try (NDManager manager = NDManager.newBaseManager()) {
            // test 3D C, H, W
            NDArray image = manager.ones(new Shape(3, 4, 2));
            float[] mean = {0.3f, 0.4f, 0.5f};
            float[] std = {0.8f, 0.8f, 0.8f};
            NDArray normalized = NDImageUtils.normalize(image, mean, std);
            NDArray actual = manager.create(new Shape(3, 4, 2));
            actual.set(new NDIndex("0,:,:"), 0.875);
            actual.set(new NDIndex("1,:,:"), 0.75f);
            actual.set(new NDIndex("2,:,:"), 0.625f);
            Assertions.assertAlmostEquals(actual, normalized);

            // test 4D N, C, H, W
            NDArray batchImages = manager.ones(new Shape(5, 3, 4, 2));
            mean = new float[] {0.6f, 0.7f, 0.8f};
            std = new float[] {0.1f, 0.2f, 0.3f};
            normalized = NDImageUtils.normalize(batchImages, mean, std);
            actual = manager.create(new Shape(5, 3, 4, 2));
            actual.set(new NDIndex(":,0,:,:"), 3.999999f);
            actual.set(new NDIndex(":,1,:,:"), 1.5);
            actual.set(new NDIndex(":,2,:,:"), 0.666666f);
            Assertions.assertAlmostEquals(actual, normalized);

            // test zero-dim
            image = manager.create(new Shape(3, 0, 1));
            normalized = NDImageUtils.normalize(image, mean, std);
            Assert.assertEquals(image, normalized);
            batchImages = manager.create(new Shape(4, 3, 0, 1));
            normalized = NDImageUtils.normalize(batchImages, mean, std);
            Assert.assertEquals(batchImages, normalized);
        }
    }

    @Test
    public void testToTensor() {
        try (NDManager manager = NDManager.newBaseManager()) {
            // test 3D C, H, W
            NDArray image = manager.randomUniform(0, 255, new Shape(4, 2, 3));
            NDArray toTensor = NDImageUtils.toTensor(image);
            NDArray actual = image.div(255f).transpose(2, 0, 1);
            Assertions.assertAlmostEquals(actual, toTensor);

            // test 4D N, C, H, W
            NDArray batchImages = manager.randomUniform(0, 255, new Shape(5, 3, 4, 2));
            toTensor = NDImageUtils.toTensor(batchImages);
            actual = batchImages.div(255f).transpose(0, 3, 1, 2);
            Assertions.assertAlmostEquals(actual, toTensor);

            // test zero-dim
            image = manager.create(new Shape(0, 1, 3));
            toTensor = NDImageUtils.toTensor(image);
            actual = manager.create(new Shape(3, 0, 1));
            Assert.assertEquals(actual, toTensor);
            batchImages = manager.create(new Shape(4, 0, 1, 3));
            toTensor = NDImageUtils.toTensor(batchImages);
            actual = manager.create(new Shape(4, 3, 0, 1));
            Assert.assertEquals(actual, toTensor);
        }
    }

    @Test(expectedExceptions = IllegalArgumentException.class)
    public void testResize() {
        try (NDManager manager = NDManager.newBaseManager()) {
            // test 3D H, W, C
            NDArray image = manager.ones(new Shape(100, 50, 3));
            NDArray result = NDImageUtils.resize(image, 50);
            NDArray actual = manager.ones(new Shape(50, 50, 3));
            Assertions.assertAlmostEquals(actual, result);
            result = NDImageUtils.resize(image, 25, 50);
            actual = manager.ones(new Shape(50, 25, 3));
            Assertions.assertAlmostEquals(actual, result);

            // test 4D N, H, W, C
            NDArray batchImages = manager.ones(new Shape(5, 75, 40, 3));
            result = NDImageUtils.resize(batchImages, 20);
            actual = manager.ones(new Shape(5, 20, 20, 3));
            Assertions.assertAlmostEquals(actual, result);
            result = NDImageUtils.resize(batchImages, 25, 50);
            actual = manager.ones(new Shape(5, 50, 25, 3));
            Assertions.assertAlmostEquals(actual, result);

            // test zero-dim
            image = manager.create(new Shape(0, 2, 3));
            // throw IllegalArgumentException
            result = NDImageUtils.resize(image, 20);
            batchImages = manager.create(new Shape(5, 0, 1, 3));
            // throw IllegalArgumentException
            result = NDImageUtils.resize(batchImages, 20);
        }
    }

    @Test
    public void testCrop() {
        try (NDManager manager = NDManager.newBaseManager()) {
            // test 3D H, W, C
            NDArray image = manager.randomUniform(0, 255, new Shape(100, 50, 3));
            NDArray result = NDImageUtils.crop(image, 10, 20, 30, 40);
            NDArray actual = image.get("20:60,10:40,:");
            Assertions.assertAlmostEquals(actual, result);

            // test 4D N, H, W, C
            NDArray batchImages = manager.randomUniform(0, 255, new Shape(5, 75, 40, 3));
            result = NDImageUtils.crop(batchImages, 5, 10, 15, 20);
            actual = batchImages.get(":,10:30,5:20,:");
            Assertions.assertAlmostEquals(actual, result);

            // test zero-dim
            image = manager.create(new Shape(0, 100, 50, 3));
            result = NDImageUtils.crop(image, 10, 20, 10, 20);
            actual = manager.create(new Shape(0, 20, 10, 3));
            Assert.assertEquals(actual, result);
        }
    }
}
