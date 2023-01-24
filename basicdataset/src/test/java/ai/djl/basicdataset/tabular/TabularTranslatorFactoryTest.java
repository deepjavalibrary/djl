/*
 * Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.basicdataset.tabular;

import ai.djl.Model;
import ai.djl.modality.Classifications;
import ai.djl.modality.Output;
import ai.djl.modality.cv.Image;
import ai.djl.translate.BasicTranslator;
import ai.djl.translate.Translator;

import org.testng.Assert;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

import java.util.HashMap;
import java.util.Map;

public class TabularTranslatorFactoryTest {
    private TabularTranslatorFactory factory;

    @BeforeClass
    public void setUp() {
        factory = new TabularTranslatorFactory();
    }

    @Test
    public void testGetSupportedTypes() {
        Assert.assertEquals(factory.getSupportedTypes().size(), 6);
    }

    // TODO: This test requires implementing the TabularTranslator constructor from arguments
    @Test(enabled = false)
    public void testNewInstance() {
        Map<String, String> arguments = new HashMap<>();
        try (Model model = Model.newInstance("test")) {
            Translator<ListFeatures, TabularResults> translator1 =
                    factory.newInstance(ListFeatures.class, TabularResults.class, model, arguments);
            Assert.assertTrue(translator1 instanceof TabularTranslator);

            Translator<MapFeatures, Classifications> translator2 =
                    factory.newInstance(MapFeatures.class, Classifications.class, model, arguments);
            Assert.assertTrue(translator2 instanceof BasicTranslator);

            Translator<ListFeatures, Float> translator3 =
                    factory.newInstance(ListFeatures.class, Float.class, model, arguments);
            Assert.assertTrue(translator3 instanceof BasicTranslator);

            Assert.assertThrows(
                    IllegalArgumentException.class,
                    () -> factory.newInstance(Image.class, Output.class, model, arguments));
        }
    }
}
