/*
 * Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.translate;

import ai.djl.ModelException;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.nn.Blocks;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;

public class DeferredTranslatorFactoryTest {

    @Test
    public void testDeferredTranslatorFactory() throws ModelException, IOException {
        DeferredTranslatorFactory factory = new DeferredTranslatorFactory();
        Assert.assertTrue(factory.getSupportedTypes().isEmpty());

        Path modelPath = Paths.get("src/test/resources/identity");
        Criteria<Image, Classifications> criteria =
                Criteria.builder()
                        .setTypes(Image.class, Classifications.class)
                        .optModelPath(modelPath)
                        .optBlock(Blocks.identityBlock())
                        .optTranslatorFactory(factory)
                        .build();

        try (ZooModel<Image, Classifications> model = criteria.loadModel()) {
            Assert.assertNotNull(model);
        }

        Criteria<Image, Classifications> criteria1 =
                Criteria.builder()
                        .setTypes(Image.class, Classifications.class)
                        .optModelPath(modelPath)
                        .optBlock(Blocks.identityBlock())
                        .optTranslatorFactory(factory)
                        .optArgument("translatorFactory", "")
                        .build();

        Assert.assertThrows(criteria1::loadModel);

        Criteria<Image, Classifications> criteria2 =
                Criteria.builder()
                        .setTypes(Image.class, Classifications.class)
                        .optModelPath(modelPath)
                        .optBlock(Blocks.identityBlock())
                        .optTranslatorFactory(factory)
                        .optArgument("translatorFactory", "not-exists")
                        .build();

        Assert.assertThrows(criteria2::loadModel);

        Criteria<Image, Classifications> criteria3 =
                Criteria.builder()
                        .setTypes(Image.class, Classifications.class)
                        .optModelPath(modelPath)
                        .optBlock(Blocks.identityBlock())
                        .optTranslatorFactory(factory)
                        .optArgument(
                                "translatorFactory",
                                "ai.djl.modality.cv.translator.StyleTransferTranslatorFactory")
                        .build();

        Assert.assertThrows(criteria3::loadModel);
    }
}
