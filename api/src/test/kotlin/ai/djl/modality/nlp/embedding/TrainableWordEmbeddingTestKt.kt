/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.modality.nlp.embedding

import ai.djl.modality.nlp.preprocess.SimpleTokenizer
import ai.djl.ndarray.NDManager
import defaultVocabulary
import org.testng.Assert
import org.testng.annotations.Test
import trainableWordEmbedding

class TrainableWordEmbeddingTestKt {
    @Test
    fun testWordEmbedding() {
        val trainableWordEmbedding = trainableWordEmbedding {
            vocabulary = defaultVocabulary {
                add(SimpleTokenizer().tokenize(TEST_STRING))
                maxTokens = 10
                unknownToken()
            }
            useDefault = true
        }
        NDManager.newBaseManager().use { manager ->
            var index = trainableWordEmbedding.preprocessWordToEmbed("Java")
            var word = trainableWordEmbedding.unembedWord(manager.create(index))
            Assert.assertEquals(word, "Java")

            index = trainableWordEmbedding.preprocessWordToEmbed(UNKNOWN_TOKEN)
            word = trainableWordEmbedding.unembedWord(manager.create(index))
            Assert.assertEquals(word, "<unk>")
        }
    }

    companion object {
        private const val TEST_STRING = "Deep Java Library (DJL) is an open-source, high-level, engine-agnostic Java framework" +
                                        " for deep learning. DJL is designed to be easy to get started with and simple" +
                                        " to\nuse for Java developers. DJL provides a native Java development experience and" +
                                        " functions like any other regular Java library.\n"
        private const val UNKNOWN_TOKEN = "UNKNOWN_TOKEN"
    }
}
