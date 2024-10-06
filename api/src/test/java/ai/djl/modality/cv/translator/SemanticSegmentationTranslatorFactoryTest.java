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
package ai.djl.modality.cv.translator;

import ai.djl.Model;
import ai.djl.modality.Input;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.output.CategoryMask;
import ai.djl.translate.Translator;

import org.testng.Assert;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

import java.util.HashMap;
import java.util.Map;

public class SemanticSegmentationTranslatorFactoryTest {

    private SemanticSegmentationTranslatorFactory factory;

    @BeforeClass
    public void setUp() {
        factory = new SemanticSegmentationTranslatorFactory();
    }

    @Test
    public void testGetSupportedTypes() {
        Assert.assertEquals(factory.getSupportedTypes().size(), 6);
    }

    @Test
    public void testNewInstance() {
        Map<String, String> arguments = new HashMap<>();
        try (Model model = Model.newInstance("test")) {
            Translator<Image, CategoryMask> translator =
                    factory.newInstance(Image.class, CategoryMask.class, model, arguments);
            Assert.assertTrue(translator instanceof SemanticSegmentationTranslator);
            Assert.assertThrows(
                    IllegalArgumentException.class,
                    () -> factory.newInstance(Input.class, Image.class, model, arguments));
        }
    }
}
