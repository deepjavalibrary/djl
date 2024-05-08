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
import ai.djl.modality.cv.output.Joints
import invoke
import org.testng.Assert
import org.testng.annotations.BeforeClass
import org.testng.annotations.Test

class SimplePoseTranslatorFactoryTestKt {
    private lateinit var factory: SimplePoseTranslatorFactory

    @BeforeClass
    fun setUp() {
        factory = SimplePoseTranslatorFactory()
    }

    @Test
    fun testGetSupportedTypes() {
        Assert.assertEquals(factory.supportedTypes.size, 2)
    }

    @Test
    fun testNewInstance() {
        val arguments: Map<String, String> = HashMap()
        Model.newInstance("test").use {
            val translator1 = factory<Image, Joints>(arguments)
            Assert.assertTrue(translator1 is SimplePoseTranslator)

            val translator2 = factory<Input, Output>(arguments)
            Assert.assertTrue(translator2 is ImageServingTranslator)
            Assert.assertThrows(IllegalArgumentException::class.java) { factory<Image, Output>(arguments) }
        }
    }
}
