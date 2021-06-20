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
import ai.djl.engine.Engine;
import ai.djl.modality.cv.Image;
import ai.djl.translate.TranslateException;
import java.io.File;
import java.io.IOException;
import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.Test;

public class BigGANTest {

    @Test
    public void testBigGAN() throws ModelException, TranslateException, IOException {
        if (!"PyTorch".equals(Engine.getInstance().getEngineName())) {
            throw new SkipException("Only works for PyTorch engine.");
        }

        Image[] generatedImages = BigGAN.generate();
        Assert.assertEquals(generatedImages.length, 5);
        Assert.assertEquals(new File("build/output/gan/").list().length, 5);

        for (Image img : generatedImages) {
            Assert.assertEquals(img.getWidth(), 128);
            Assert.assertEquals(img.getHeight(), 128);
        }
    }
}
