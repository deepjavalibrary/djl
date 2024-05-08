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
import newInstance
import org.testng.Assert
import org.testng.annotations.BeforeClass
import org.testng.annotations.Test

class BigGANTranslatorFactoryTestKt {
    private lateinit var factory: BigGANTranslatorFactory

    @BeforeClass
    fun setUp() {
        factory = BigGANTranslatorFactory()
    }

    @Test
    fun testGetSupportedTypes() {
        Assert.assertEquals(factory.supportedTypes.size, 1)
    }

    @Test
    fun testNewInstance() {
        val arguments: MutableMap<String, Float> = HashMap()
        arguments["truncation"] = 0.6f
        Model.newInstance("test").use { model ->
            val translator = factory.newInstance<IntArray, Array<Image>>(model, arguments)
            Assert.assertTrue(translator is BigGANTranslator)
            Assert.assertThrows(IllegalArgumentException::class.java) {
                factory.newInstance(Input::class.java, Output::class.java, model, arguments)
            }
        }
    }
}
