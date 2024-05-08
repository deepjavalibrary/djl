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
package ai.djl.modality.cv.translator

import ai.djl.Model
import ai.djl.modality.Input
import ai.djl.modality.Output
import ai.djl.modality.cv.Image
import ai.djl.modality.cv.output.DetectedObjects
import ai.djl.translate.BasicTranslator
import invoke
import org.testng.Assert
import org.testng.annotations.BeforeClass
import org.testng.annotations.Test
import java.io.InputStream
import java.net.URL
import java.nio.file.Path

class YoloTranslatorFactoryTestKt {
    private lateinit var factory: YoloTranslatorFactory

    @BeforeClass
    fun setUp() {
        factory = YoloTranslatorFactory()
    }

    @Test
    fun testGetSupportedTypes() {
        Assert.assertEquals(factory.supportedTypes.size, 5)
    }

    @Test fun testNewInstance() {
        val arguments: Map<String, String> = HashMap()
        Model.newInstance("test").use { model ->
            val translator1 = factory<Image, DetectedObjects>(model, arguments)
            Assert.assertTrue(translator1 is YoloTranslator)

            val translator2 = factory<Path, DetectedObjects>(model, arguments)
            Assert.assertTrue(translator2 is BasicTranslator<*, *>)

            val translator3 = factory<URL, DetectedObjects>(model, arguments)
            Assert.assertTrue(translator3 is BasicTranslator<*, *>)

            val translator4 = factory<InputStream, DetectedObjects>(model, arguments)
            Assert.assertTrue(translator4 is BasicTranslator<*, *>)

            val translator5 = factory<Input, Output>(model, arguments)
            Assert.assertTrue(translator5 is ImageServingTranslator)
            Assert.assertThrows(IllegalArgumentException::class.java) { factory<Image, Output>(model, arguments) }
        }
    }
}
