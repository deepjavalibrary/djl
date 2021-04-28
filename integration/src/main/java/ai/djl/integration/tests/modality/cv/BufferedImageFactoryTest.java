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
package ai.djl.integration.tests.modality.cv;

import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.testing.Assertions;
import java.io.IOException;
import org.testng.Assert;
import org.testng.annotations.Test;

public class BufferedImageFactoryTest {
    @Test
    public void testLoadImage() throws IOException {
        try (NDManager manager = NDManager.newBaseManager()) {
            ImageFactory factory = ImageFactory.getInstance();
            Image img =
                    factory.fromUrl(
                            "https://github.com/deepjavalibrary/djl/raw/master/examples/src/test/resources/dog_bike_car.jpg");
            NDArray array = img.toNDArray(manager);
            Assert.assertEquals(new Shape(img.getHeight(), img.getWidth(), 3), array.getShape());
        }
    }

    @Test
    public void testFormNDArray() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array = manager.arange(0.0f, 12.0f).reshape(3, 2, 2);
            ImageFactory factory = ImageFactory.getInstance();
            Image image = factory.fromNDArray(array.toType(DataType.INT8, true));
            NDArray converted =
                    image.toNDArray(manager).transpose(2, 0, 1).toType(DataType.FLOAT32, true);
            Assertions.assertAlmostEquals(array, converted);
        }
    }
}
