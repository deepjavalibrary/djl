/*
 * Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.examples.inference;

import ai.djl.ModelException;
import ai.djl.examples.inference.sr.SuperResolution;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.translate.TranslateException;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.Test;

public class SuperResolutionTest {

    @Test
    public void testSuperResolution() throws ModelException, TranslateException, IOException {
        String imagePath = "src/test/resources/";
        Image fox = ImageFactory.getInstance().fromFile(Paths.get(imagePath + "fox.png"));
        List<Image> inputImages = Arrays.asList(fox, fox);

        List<Image> enhancedImages = SuperResolution.enhance(inputImages);

        if (enhancedImages == null) {
            throw new SkipException("Only works for TensorFlow engine.");
        }

        Assert.assertEquals(enhancedImages.size(), 2);
        int size = 4 * fox.getWidth();
        for (Image img : enhancedImages) {
            Assert.assertEquals(img.getWidth(), size);
            Assert.assertEquals(img.getHeight(), size);
        }
    }
}
