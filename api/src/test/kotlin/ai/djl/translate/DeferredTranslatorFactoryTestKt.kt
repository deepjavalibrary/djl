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
package ai.djl.translate

import ai.djl.ModelException
import ai.djl.modality.Classifications
import ai.djl.modality.cv.Image
import ai.djl.nn.Blocks
import ai.djl.repository.zoo.Criteria
import criteria
import org.testng.Assert
import org.testng.annotations.Test
import java.io.IOException
import java.nio.file.Paths

class DeferredTranslatorFactoryTestKt {
    @Test @Throws(ModelException::class, IOException::class)
    fun testDeferredTranslatorFactory() {
        val factory = DeferredTranslatorFactory()
        Assert.assertTrue(factory.supportedTypes.isEmpty())

        val path = Paths.get("src/test/resources/identity")
        val criteria = criteria<Image, Classifications> {
            modelPath = path
            block = Blocks.identityBlock()
            translatorFactory = factory
        }

        criteria.loadModel().use(Assert::assertNotNull)
        val criteria1 = criteria<Image, Classifications> {
            modelPath = path
            block = Blocks.identityBlock()
            translatorFactory = factory
            arguments("translatorFactory" to "")
        }

        Assert.assertThrows { criteria1.loadModel() }

        val criteria2 = criteria<Image, Classifications> {
            modelPath = path
            block = Blocks.identityBlock()
            translatorFactory = factory
            arguments("translatorFactory" to "not-exists")
        }

        Assert.assertThrows { criteria2.loadModel() }

        val criteria3 = criteria<Image, Classifications> {
            modelPath = path
            block = Blocks.identityBlock()
            translatorFactory = factory
            arguments("translatorFactory" to "ai.djl.modality.cv.translator.StyleTransferTranslatorFactory")
        }

        Assert.assertThrows { criteria3.loadModel() }
    }
}
