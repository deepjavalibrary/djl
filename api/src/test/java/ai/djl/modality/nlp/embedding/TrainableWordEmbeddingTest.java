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
package ai.djl.modality.nlp.embedding;

import ai.djl.modality.nlp.DefaultVocabulary;
import ai.djl.modality.nlp.preprocess.SimpleTokenizer;
import ai.djl.ndarray.NDManager;
import org.testng.Assert;
import org.testng.annotations.Test;

public class TrainableWordEmbeddingTest {
    private static final String TEST_STRING =
            "Deep Java Library (DJL) is an open-source, high-level, engine-agnostic Java framework for deep learning. DJL is designed to be easy to get started with and simple to\n"
                    + "use for Java developers. DJL provides a native Java development experience and functions like any other regular Java library.\n";
    private static final String UNKNOWN_TOKEN = "UNKNOWN_TOKEN";

    @Test
    public void testWordEmbedding() {
        TrainableWordEmbedding trainableWordEmbedding =
                TrainableWordEmbedding.builder()
                        .setVocabulary(
                                DefaultVocabulary.builder()
                                        .add(new SimpleTokenizer().tokenize(TEST_STRING))
                                        .optMaxTokens(10)
                                        .optUnknownToken()
                                        .build())
                        .optUseDefault(true)
                        .build();
        try (NDManager manager = NDManager.newBaseManager()) {
            long index = trainableWordEmbedding.preprocessWordToEmbed("Java");
            String word = trainableWordEmbedding.unembedWord(manager.create(index));
            Assert.assertEquals(word, "Java");

            index = trainableWordEmbedding.preprocessWordToEmbed(UNKNOWN_TOKEN);
            word = trainableWordEmbedding.unembedWord(manager.create(index));
            Assert.assertEquals(word, "<unk>");
        }
    }
}
