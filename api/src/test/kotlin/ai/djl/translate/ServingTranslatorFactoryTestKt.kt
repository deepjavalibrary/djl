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
package ai.djl.translate

import ai.djl.ModelException
import ai.djl.modality.Input
import ai.djl.modality.Output
import ai.djl.nn.Blocks
import ai.djl.repository.zoo.Criteria
import ai.djl.translate.TranslateException
import criteria
import org.testng.Assert
import org.testng.annotations.Test
import java.io.IOException
import java.nio.file.Files
import java.nio.file.Paths

class ServingTranslatorFactoryTestKt {
    @Test @Throws(IOException::class, TranslateException::class, ModelException::class)
    fun test() {
        val path = Paths.get("build/model")
        Files.createDirectories(path)
        var criteria = criteria<Input, Output> {
            modelPath = path
            modelName = "identity"
            arguments("application" to "image_classification")
            options("hasParameter" to "false")
            block = Blocks.identityBlock()
        }

        criteria.loadModel().use { model ->
            val className = model.translator.javaClass.simpleName
            Assert.assertEquals(className, "ImageServingTranslator")
        }
        criteria = criteria<Input, Output> {
            modelPath = path
            modelName = "identity"
            // HF style task name
            arguments("task" to "fill-mask")
            options("hasParameter" to "false")
            block = Blocks.identityBlock()
        }

        criteria.loadModel().use { model ->
            val className = model.translator.javaClass.simpleName
            // Tokenizer is not in class path fallback to NoopServingTranslator
            Assert.assertEquals(className, "NoopServingTranslator")
        }
        criteria = criteria<Input, Output> {
            modelPath = path
            modelName = "identity"
            // HF style task name
            arguments("task" to "question-answering")
            options("hasParameter" to "false")
            block = Blocks.identityBlock()
        }

        criteria.loadModel().use { model ->
            val className = model.translator.javaClass.simpleName
            // Tokenizer is not in class path fallback to NoopServingTranslator
            Assert.assertEquals(className, "NoopServingTranslator")
        }
        criteria = criteria<Input, Output> {
            modelPath = path
            modelName = "identity"
            // HF style task name
            arguments("task" to "sentence-similarity")
            options("hasParameter" to "false")
            block = Blocks.identityBlock()
        }

        criteria.loadModel().use { model ->
            val className = model.translator.javaClass.simpleName
            // Tokenizer is not in class path fallback to NoopServingTranslator
            Assert.assertEquals(className, "NoopServingTranslator")
        }
        criteria = criteria<Input, Output> {
            modelPath = path
            modelName = "identity"
            // HF style task name
            arguments("task" to "text-classification")
            options("hasParameter" to "false")
            block = Blocks.identityBlock()
        }

        criteria.loadModel().use { model ->
            val className = model.translator.javaClass.simpleName
            // Tokenizer is not in class path fallback to NoopServingTranslator
            Assert.assertEquals(className, "NoopServingTranslator")
        }
        criteria = criteria<Input, Output> {
            modelPath = path
            modelName = "identity" // HF style task name
            arguments("task" to "token-classification")
            options("hasParameter" to "false")
            block = Blocks.identityBlock()
        }

        criteria.loadModel().use { model ->
            val className = model.translator.javaClass.simpleName
            // Tokenizer is not in class path fallback to NoopServingTranslator
            Assert.assertEquals(className, "NoopServingTranslator")
        }
    }
}
