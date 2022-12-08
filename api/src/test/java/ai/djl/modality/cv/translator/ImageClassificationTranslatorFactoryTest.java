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
import ai.djl.modality.Classifications;
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.modality.cv.Image;
import ai.djl.translate.BasicTranslator;
import ai.djl.translate.Translator;

import org.testng.Assert;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

import java.io.InputStream;
import java.net.URL;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;

public class ImageClassificationTranslatorFactoryTest {

    private ImageClassificationTranslatorFactory factory;

    @BeforeClass
    public void setUp() {
        factory = new ImageClassificationTranslatorFactory();
    }

    @Test
    public void testGetSupportedTypes() {
        Assert.assertEquals(factory.getSupportedTypes().size(), 5);
    }

    @Test
    public void testNewInstance() {
        Map<String, String> arguments = new HashMap<>();
        try (Model model = Model.newInstance("test")) {
            Translator<Image, Classifications> translator1 =
                    factory.newInstance(Image.class, Classifications.class, model, arguments);
            Assert.assertTrue(translator1 instanceof ImageClassificationTranslator);

            Translator<Path, Classifications> translator2 =
                    factory.newInstance(Path.class, Classifications.class, model, arguments);
            Assert.assertTrue(translator2 instanceof BasicTranslator);

            Translator<URL, Classifications> translator3 =
                    factory.newInstance(URL.class, Classifications.class, model, arguments);
            Assert.assertTrue(translator3 instanceof BasicTranslator);

            Translator<InputStream, Classifications> translator4 =
                    factory.newInstance(InputStream.class, Classifications.class, model, arguments);
            Assert.assertTrue(translator4 instanceof BasicTranslator);

            Translator<Input, Output> translator5 =
                    factory.newInstance(Input.class, Output.class, model, arguments);
            Assert.assertTrue(translator5 instanceof ImageServingTranslator);

            Assert.assertThrows(
                    IllegalArgumentException.class,
                    () -> factory.newInstance(Image.class, Output.class, model, arguments));
        }
    }
}
