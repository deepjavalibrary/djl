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
import ai.djl.modality.Input
import ai.djl.modality.Output
import ai.djl.nn.Blocks
import criteria
import org.testng.Assert
import org.testng.annotations.Test
import java.io.IOException
import java.nio.file.Paths

class NoopServingTranslatorFactoryTestKt {
    @Test @Throws(ModelException::class, IOException::class)
    fun testNoopTranslatorFactory() {
        val factory = NoopServingTranslatorFactory()
        Assert.assertEquals(factory.supportedTypes.size, 1)

        val criteria = criteria<Input, Output> {
            modelPath = Paths.get("src/test/resources/identity")
            block = Blocks.identityBlock()
            translatorFactory = factory
        }

        criteria.loadModel().use(Assert::assertNotNull)
    }
}
