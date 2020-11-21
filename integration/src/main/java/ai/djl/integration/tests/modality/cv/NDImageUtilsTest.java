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

import ai.djl.engine.Engine;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.Shape;
import ai.djl.testing.Assertions;
import ai.djl.util.cuda.CudaUtils;
import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.Test;

public class NDImageUtilsTest {

    @Test
    public void testNormalize() {
        if ("TensorFlow".equals(Engine.getInstance().getEngineName())) {
            throw new SkipException("TensorFlow use channels last by default.");
        }
        try (NDManager manager = NDManager.newBaseManager()) {
            // test 3D C, H, W
            NDArray image = manager.ones(new Shape(3, 4, 2));
            float[] mean = {0.3f, 0.4f, 0.5f};
            float[] std = {0.8f, 0.8f, 0.8f};
            NDArray normalized = NDImageUtils.normalize(image, mean, std);
            NDArray expected = manager.create(new Shape(3, 4, 2));
            expected.set(new NDIndex("0,:,:"), 0.875);
            expected.set(new NDIndex("1,:,:"), 0.75f);
            expected.set(new NDIndex("2,:,:"), 0.625f);
            Assertions.assertAlmostEquals(normalized, expected);

            // test 4D N, C, H, W
            NDArray batchImages = manager.ones(new Shape(5, 3, 4, 2));
            mean = new float[] {0.6f, 0.7f, 0.8f};
            std = new float[] {0.1f, 0.2f, 0.3f};
            normalized = NDImageUtils.normalize(batchImages, mean, std);
            expected = manager.create(new Shape(5, 3, 4, 2));
            expected.set(new NDIndex(":,0,:,:"), 3.999999f);
            expected.set(new NDIndex(":,1,:,:"), 1.5);
            expected.set(new NDIndex(":,2,:,:"), 0.666666f);
            Assertions.assertAlmostEquals(normalized, expected);

            // test zero-dim
            image = manager.create(new Shape(3, 0, 1));
            normalized = NDImageUtils.normalize(image, mean, std);
            Assert.assertEquals(normalized, image);
            batchImages = manager.create(new Shape(4, 3, 0, 1));
            normalized = NDImageUtils.normalize(batchImages, mean, std);
            Assert.assertEquals(normalized, batchImages);
        }
    }

    @Test
    public void testToTensor() {
        if ("TensorFlow".equals(Engine.getInstance().getEngineName())) {
            throw new SkipException("TensorFlow use channels last by default.");
        }
        try (NDManager manager = NDManager.newBaseManager()) {
            // test 3D C, H, W
            NDArray image = manager.randomUniform(0, 255, new Shape(4, 2, 3));
            NDArray toTensor = NDImageUtils.toTensor(image);
            NDArray expected = image.div(255f).transpose(2, 0, 1);
            Assertions.assertAlmostEquals(toTensor, expected);

            // test 4D N, C, H, W
            NDArray batchImages = manager.randomUniform(0, 255, new Shape(5, 3, 4, 2));
            toTensor = NDImageUtils.toTensor(batchImages);
            expected = batchImages.div(255f).transpose(0, 3, 1, 2);
            Assertions.assertAlmostEquals(toTensor, expected);

            // test zero-dim
            image = manager.create(new Shape(0, 1, 3));
            toTensor = NDImageUtils.toTensor(image);
            expected = manager.create(new Shape(3, 0, 1));
            Assert.assertEquals(toTensor, expected);
            batchImages = manager.create(new Shape(4, 0, 1, 3));
            toTensor = NDImageUtils.toTensor(batchImages);
            expected = manager.create(new Shape(4, 3, 0, 1));
            Assert.assertEquals(toTensor, expected);
        }
    }

    @Test
    public void testResize() {
        try (NDManager manager = NDManager.newBaseManager()) {
            Image.Interpolation[] interpolations = {
                Image.Interpolation.NEAREST,
                Image.Interpolation.BILINEAR,
                Image.Interpolation.AREA,
                Image.Interpolation.BICUBIC
            };
            for (Image.Interpolation interpolation : interpolations) {
                // mxnet only support BILINEAR on GPU
                if (CudaUtils.hasCuda() && !interpolation.equals(Image.Interpolation.BILINEAR)) {
                    continue;
                }
                // test 3D H, W, C
                NDArray image = manager.ones(new Shape(100, 50, 3));
                NDArray result = NDImageUtils.resize(image, 50, 50);
                NDArray expected = manager.ones(new Shape(50, 50, 3));
                Assertions.assertAlmostEquals(result, expected);
                result = NDImageUtils.resize(image, 25, 50, interpolation);
                expected = manager.ones(new Shape(50, 25, 3));
                Assertions.assertAlmostEquals(result, expected);

                // test 4D N, H, W, C
                NDArray batchImages = manager.ones(new Shape(5, 75, 40, 3));
                result = NDImageUtils.resize(batchImages, 20);
                expected = manager.ones(new Shape(5, 20, 20, 3));
                Assertions.assertAlmostEquals(result, expected);
                result = NDImageUtils.resize(batchImages, 25, 50, interpolation);
                expected = manager.ones(new Shape(5, 50, 25, 3));
                Assertions.assertAlmostEquals(result, expected);

                Assert.expectThrows(
                        IllegalArgumentException.class,
                        () -> {
                            // test zero-dim
                            NDArray img = manager.create(new Shape(0, 2, 3));
                            NDImageUtils.resize(img, 20);
                        });

                Assert.expectThrows(
                        IllegalArgumentException.class,
                        () -> {
                            NDArray img = manager.create(new Shape(5, 0, 1, 3));
                            NDImageUtils.resize(img, 20);
                        });
            }
        }
    }

    @Test
    public void testCrop() {
        try (NDManager manager = NDManager.newBaseManager()) {
            // test 3D H, W, C
            NDArray image = manager.randomUniform(0, 255, new Shape(100, 50, 3));
            NDArray result = NDImageUtils.crop(image, 10, 20, 30, 40);
            NDArray expected = image.get("20:60,10:40,:");
            Assertions.assertAlmostEquals(result, expected);

            // test 4D N, H, W, C
            NDArray batchImages = manager.randomUniform(0, 255, new Shape(5, 75, 40, 3));
            result = NDImageUtils.crop(batchImages, 5, 10, 15, 20);
            expected = batchImages.get(":,10:30,5:20,:");
            Assertions.assertAlmostEquals(result, expected);

            // test zero-dim
            image = manager.create(new Shape(0, 100, 50, 3));
            result = NDImageUtils.crop(image, 10, 20, 10, 20);
            expected = manager.create(new Shape(0, 20, 10, 3));
            Assert.assertEquals(result, expected);
        }
    }

    @Test
    public void testRandomCrop() {
        try (NDManager manager = NDManager.newBaseManager()) {
            // make sure runnable
            NDArray image = manager.randomUniform(0, 255, new Shape(100, 50, 3));
            NDImageUtils.randomResizedCrop(image, 10, 10, 0.3, 0.7, 0.5, 2);
        }
    }

    @Test
    public void testRandomFlip() {
        try (NDManager manager = NDManager.newBaseManager()) {
            manager.getEngine().setRandomSeed(1234);
            // flip lr
            NDArray image = manager.randomUniform(0, 255, new Shape(10, 10, 3));
            NDArray result = NDImageUtils.randomFlipLeftRight(image);
            Assert.assertEquals(result.get("0,:,0"), image.get("0,:,0").flip(0));
            // fliptb
            result = NDImageUtils.randomFlipTopBottom(image);
            Assert.assertEquals(result.get(":,0,0"), image.get(":,0,0").flip(0));
        }
    }

    @Test
    public void testRandomColor() {
        try (NDManager manager = NDManager.newBaseManager()) {
            // make sure runnable
            NDArray image = manager.randomUniform(0, 255, new Shape(10, 10, 3));
            NDImageUtils.randomBrightness(image, 0.6f);
            NDImageUtils.randomHue(image, 0.6f);
            NDImageUtils.randomColorJitter(image, 0.6f, 0.8f, 0.7f, 0.6f);
        }
    }
}
