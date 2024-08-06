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
package ai.djl.examples.inference.cv;

import ai.djl.ModelException;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.testing.TestRequirements;
import ai.djl.translate.TranslateException;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.IOException;
import java.nio.file.Paths;

public class StyleTransferTest {

    @Test
    public void testStyleTransfer() throws IOException, ModelException, TranslateException {
        TestRequirements.linux();

        String imagePath = "src/test/resources/mountains.png";
        Image input = ImageFactory.getInstance().fromFile(Paths.get(imagePath));
        Image generatedImage = StyleTransfer.transfer(input, StyleTransfer.Artist.MONET);

        Assert.assertEquals(generatedImage.getWidth(), input.getWidth());
        Assert.assertEquals(generatedImage.getHeight(), input.getHeight());
    }
}
