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
package ai.djl.android.core;

import static androidx.test.platform.app.InstrumentationRegistry.getInstrumentation;

import static org.junit.Assert.assertEquals;

import android.content.Context;

import androidx.test.ext.junit.runners.AndroidJUnit4;

import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;

import org.junit.Test;
import org.junit.runner.RunWith;

import java.io.IOException;

/**
 * Instrumented test, which will execute on an Android device.
 *
 * @see <a href="http://d.android.com/tools/testing">Testing documentation</a>
 */
@RunWith(AndroidJUnit4.class)
public class BitmapWrapperTest {

    @Test
    public void useAppContext() {
        // Context of the app under test.
        Context appContext = getInstrumentation().getTargetContext();

        assertEquals("ai.djl.android.core.test", appContext.getPackageName());
    }

    @Test
    public void testImageToNDArray() throws IOException {
        try (NDManager manager = NDManager.newBaseManager()) {
            ImageFactory factory = ImageFactory.getInstance();
            Image img =
                    factory.fromUrl(
                            "https://github.com/deepjavalibrary/djl/raw/master/examples/src/test/resources/dog_bike_car.jpg");
            NDArray array = img.toNDArray(manager);
            assertEquals(new Shape(img.getHeight(), img.getWidth(), 3), array.getShape());
        }
    }

    @Test
    public void testImageGetSubImage() throws IOException {
        try (NDManager manager = NDManager.newBaseManager()) {
            ImageFactory factory = ImageFactory.getInstance();
            Image img =
                    factory.fromUrl(
                            "https://github.com/deepjavalibrary/djl/raw/master/examples/src/test/resources/dog_bike_car.jpg");
            NDArray array = img.toNDArray(manager);
            Image subImg = img.getSubImage(10, 20, 30, 40);
            NDArray subArray = subImg.toNDArray(manager);
            assertEquals(array.get("20:60,10:40,:"), subArray);
        }
    }

    @Test
    public void testImageDuplicate() throws IOException {
        try (NDManager manager = NDManager.newBaseManager()) {
            ImageFactory factory = ImageFactory.getInstance();
            Image img =
                    factory.fromUrl(
                            "https://github.com/deepjavalibrary/djl/raw/master/examples/src/test/resources/dog_bike_car.jpg");
            NDArray array = img.toNDArray(manager);
            NDArray duplicate = img.duplicate().toNDArray(manager);
            assertEquals(array, duplicate);
        }
    }

    @Test
    public void testImageFromNDArray() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray array = manager.arange(0.0f, 12.0f).reshape(3, 2, 2);
            ImageFactory factory = ImageFactory.getInstance();
            Image image = factory.fromNDArray(array);
            NDArray converted =
                    image.toNDArray(manager).transpose(2, 0, 1).toType(DataType.FLOAT32, true);
            assertEquals(array, converted);

            array = manager.arange(0, 12, 1, DataType.UINT8).reshape(3, 2, 2);
            array = array.transpose(1, 2, 0);
            image = factory.fromNDArray(array);
            converted = image.toNDArray(manager);
            assertEquals(array, converted);
        }
    }
}
