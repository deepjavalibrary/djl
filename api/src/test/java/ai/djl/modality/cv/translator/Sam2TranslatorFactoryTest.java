/*
 * Copyright 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import ai.djl.modality.Output;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.translator.Sam2Translator.Sam2Input;
import ai.djl.translate.Translator;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.util.HashMap;
import java.util.Map;

public class Sam2TranslatorFactoryTest {

    @Test
    public void testNewInstance() {
        Sam2TranslatorFactory factory = new Sam2TranslatorFactory();
        Assert.assertEquals(factory.getSupportedTypes().size(), 2);
        Map<String, String> arguments = new HashMap<>();
        try (Model model = Model.newInstance("test")) {
            Translator<Sam2Input, DetectedObjects> translator1 =
                    factory.newInstance(Sam2Input.class, DetectedObjects.class, model, arguments);
            Assert.assertTrue(translator1 instanceof Sam2Translator);

            Translator<Input, Output> translator5 =
                    factory.newInstance(Input.class, Output.class, model, arguments);
            Assert.assertTrue(translator5 instanceof Sam2ServingTranslator);

            Assert.assertThrows(
                    IllegalArgumentException.class,
                    () -> factory.newInstance(Image.class, Output.class, model, arguments));
        }
    }
}
